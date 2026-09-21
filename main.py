import streamlit as st
import pandas as pd
import json
from google import genai

# 2. Load and Filter Data
df_projects = pd.read_csv('data/horizon/project.csv', sep=';', on_bad_lines='skip')
df_orgs = pd.read_csv('data/horizon/organization.csv', sep=';', on_bad_lines='skip')
df_merged = pd.merge(df_orgs, df_projects, left_on='projectID', right_on='id')
df_commercial = df_merged[df_merged['activityType'] == 'PRC']
df_ai_ready = df_commercial[['name', 'city', 'country', 'title', 'objective']]

available_locations = [
    'All Europe', 'Albania', 'Andorra', 'Armenia', 'Austria', 'Azerbaijan',
    'Belarus', 'Belgium', 'Bosnia and Herzegovina', 'Bulgaria', 'Croatia',
    'Cyprus', 'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France',
    'Georgia', 'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland', 'Italy',
    'Kosovo', 'Latvia', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Malta',
    'Moldova', 'Monaco', 'Montenegro', 'Netherlands', 'North Macedonia',
    'Norway', 'Poland', 'Portugal', 'Romania', 'San Marino', 'Serbia',
    'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'Turkey',
    'Ukraine', 'United Kingdom', 'Vatican City'
]

country_mapping = {
    'Albania': 'AL', 'Andorra': 'AD', 'Armenia': 'AM', 'Austria': 'AT', 'Azerbaijan': 'AZ',
    'Belarus': 'BY', 'Belgium': 'BE', 'Bosnia and Herzegovina': 'BA', 'Bulgaria': 'BG',
    'Croatia': 'HR', 'Cyprus': 'CY', 'Czech Republic': 'CZ', 'Denmark': 'DK', 'Estonia': 'EE',
    'Finland': 'FI', 'France': 'FR', 'Georgia': 'GE', 'Germany': 'DE', 'Greece': 'EL',  # CORDIS uses EL
    'Hungary': 'HU', 'Iceland': 'IS', 'Ireland': 'IE', 'Italy': 'IT', 'Kosovo': 'XK',
    'Latvia': 'LV', 'Liechtenstein': 'LI', 'Lithuania': 'LT', 'Luxembourg': 'LU', 'Malta': 'MT',
    'Moldova': 'MD', 'Monaco': 'MC', 'Montenegro': 'ME', 'Netherlands': 'NL',
    'North Macedonia': 'MK', 'Norway': 'NO', 'Poland': 'PL', 'Portugal': 'PT', 'Romania': 'RO',
    'San Marino': 'SM', 'Serbia': 'RS', 'Slovakia': 'SK', 'Slovenia': 'SI', 'Spain': 'ES',
    'Sweden': 'SE', 'Switzerland': 'CH', 'Turkey': 'TR', 'Ukraine': 'UA',
    'United Kingdom': 'UK', 'Vatican City': 'VA'  # CORDIS uses UK
}

# 1. Give your app a title
st.title("Company Search")
st.markdown("Automatically filter Horizon Europe data to find early-stage tech ventures.")

# --- SIDEBAR LAYOUT ADDED HERE ---
with st.sidebar:
    st.header("Search Filters")

    # Create the dropdown menu inside the sidebar
    chosen_locations = st.multiselect(
        "Select target markets (leave blank for all of Europe):",
        options=available_locations,
        default=['Belgium', 'Netherlands']
    )

    st.markdown("---")
    # Move the scout button to the sidebar
    scout_button = st.button("Scout New Batch", type="primary", use_container_width=True)

if len(chosen_locations) > 0:
    # Safely translate selected names, ignoring 'All Europe' if clicked
    chosen_codes = [country_mapping[country] for country in chosen_locations if country in country_mapping]

    # Filter using the codes!
    filtered_df = df_ai_ready[df_ai_ready['country'].isin(chosen_codes)]
else:
    filtered_df = df_ai_ready


def run_ai_scout2():
    """Connects to Gemini and processes a batch of companies."""
    print("Scouting for deep-tech...")

    # --- API SECRETS ADDED HERE ---
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

    # Added a quick safety check so Pandas doesn't crash if the filter has <100 companies
    sample_size = min(100, len(filtered_df))
    if sample_size == 0:
        return pd.DataFrame()

    sample_batch = filtered_df.sample(n=sample_size)

    # Convert this batch into a readable text string for the AI
    batch_text = ""
    for index, row in sample_batch.iterrows():
        short_objective = str(row['objective'])[:400]

        batch_text += f"Company: {row['name']}\nLocation: {row['city']}, {row['country']}\nProject: {row['title']}\nDescription: {short_objective}\n---\n"

    # 4. The strict prompt telling the AI exactly what to do
    prompt = f"""
    You are an expert scouting analyst looking for early-stage frontier technology startups and spin-offs preferably pre-seed or seed stage. 

    Here is a batch of raw European grant projects:
    {batch_text}

    EVALUATION CRITERIA:
    1. THE VIBE & STAGE (MUST MATCH):
       - We are looking for early-stage innovators, university spin-offs, and emerging ventures bringing highly defensible, proprietary technology to market (e.g., prototyping, lab-to-market, or early commercial deployment).
       - They must possess deep technical defensibility or fundamental innovation (e.g., novel LiDAR systems, fundamental AI/algorithmic breakthroughs, advanced robotics, new physical devices, or engineered biosystems). 
    2. EXCLUSIONS (DO NOT INCLUDE):
       - Established multinational conglomerates or legacy manufacturers participating in standard consortium tasks.
       - Pure commodity software, standard SaaS apps, web dashboards, digital marketplaces, or basic IT infrastructure.
       - Standard service providers, consulting firms, testing labs, or generic component assembly without unique IP.

    For each commercial company that strictly meets this startup/frontier tech profile, extract the information as a clean JSON array of objects.
    Each object must have exactly these keys:
    - "Company": The legal or operating company name
    - "Product": A concrete description of their core proprietary technology or product (e.g., "Autonomous LiDAR system", "Photonic laser chip", "Next-gen routing algorithm")
    - "Sector": Short 1-3 word category (e.g., "Autonomous Vehicles", "Advanced Materials", "AI Infrastructure")
    - "Location": City and country
    - "Size": Employee count or range if explicitly mentioned, otherwise -1

    Return ONLY the raw JSON array. Do not include markdown code blocks, backticks, or intro/outro explanations.
    """

    print("Sending batch to Gemini to extract startups...")

    # 5. Define your fallback list of models (fastest/cheapest first)
    fallback_models = [
        'gemini-3.5-flash',
        'gemini-3.6-flash',
        'gemini-3.8-flash'
    ]


    # 6. Loop through the models
    for model_name in fallback_models:
        try:
            print(f"Attempting extraction with {model_name}...")

            # Call the API
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )

            # --- MARKDOWN STRIPPING ADDED HERE ---
            # This prevents the app from crashing if Gemini adds ```json
            clean_text = response.text.strip().replace("```json", "").replace("```", "")
            clean_data = json.loads(clean_text)

            print(f"Success! Model {model_name} extracted the data.")
            return pd.DataFrame(clean_data)

        except Exception as e:
            # If it fails (503 error, JSON parse error, etc.), print a warning and continue the loop
            print(f"Warning: {model_name} failed. Moving to next model. Error details: {e}")
            continue

    # 7. If the loop finishes all models and still hasn't returned anything:
    print("Error: All models are currently unavailable or failed to parse.")
    return pd.DataFrame()  # Return an empty dataframe on failure


def show_ready_dataframe(data_to_be_shown, tb_removed="no_drops"):
    if tb_removed != "no_drops" and tb_removed in data_to_be_shown.columns:
        manipulated_data = data_to_be_shown.drop(columns=[tb_removed])
        st.dataframe(manipulated_data)
    else:
        st.dataframe(data_to_be_shown)


# Initialize Streamlit memory (Session State) so data survives dropdown clicks
if "scouted_data" not in st.session_state:
    empty_data = {"Company": [], "Product": [], "Sector": [], "Location": [], "Size": []}
    st.session_state["scouted_data"] = pd.DataFrame(empty_data)

# 2. The Scout Button (now connected to the sidebar button)
if scout_button:
    new_data_df = run_ai_scout2()

    if not new_data_df.empty:
        st.success(f"Successfully scouted {len(new_data_df)} companies!")
        # Save to memory instead of a temporary variable
        st.session_state["scouted_data"] = pd.DataFrame(new_data_df)
    else:
        st.error("AI failed to find match.")

# 3. Pull the dataframe from memory for the rest of the script
df = st.session_state["scouted_data"].copy()

# Only show filters and table if there is actual data
if not df.empty:

    st.write("Use the box below to filter companies:")

    # Dynamically load locations found by the AI
    available_locations_results = ["All"] + sorted(df["Location"].unique().tolist())
    chosen_location_result = st.selectbox("Select a Location:", available_locations_results)

    # 4. The App Logic
    if chosen_location_result == "All":
        show_ready_dataframe(df, "Size_trans")
        export_df = df
    else:
        filtered_table = df[(df["Location"] == chosen_location_result)]
        show_ready_dataframe(filtered_table, "Size_trans")
        export_df = filtered_table

    st.markdown("---")

    # --- CSV DOWNLOAD BUTTON ADDED HERE ---
    csv_data = export_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv_data,
        file_name='scouted_deep_tech_startups.csv',
        mime='text/csv',
    )

else:
    st.info("Click 'Scout New Batch' in the sidebar to begin fetching data.")

st.write("Data sourced and modified from EU CORDIS, no personal data has been used.")