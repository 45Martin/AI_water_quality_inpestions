import streamlit as st
import pandas as pd
import joblib
import numpy as np
from PIL import Image

# Set page config
st.set_page_config(
    page_title="AquaSafe Pro - Water Quality Classifier",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the trained pipeline
@st.cache_resource
def load_pipeline():
    return joblib.load('water_quality_model.pkl')

pipeline = load_pipeline()

# App title and header
st.title('💧 AquaSafe Pro Water Quality Classifier')
st.markdown("""
<div style="background-color:#f0f2f6;padding:10px;border-radius:10px;margin-bottom:20px;">
    <h3 style="color:#1e3d6b;text-align:center;">Advanced water safety analysis with expanded parameter ranges</h3>
</div>
""", unsafe_allow_html=True)

# Sidebar with info
with st.sidebar:
    st.header("About AquaSafe Pro")
    st.markdown("""
    This enhanced version includes wider parameter ranges for comprehensive water quality analysis.
    """)
    
    st.markdown("---")
    st.subheader("Safe Water Parameters")
    st.markdown("""
    - **pH:** 6.5 - 8.5 (expanded range: 0-14)
    - **Turbidity:** < 5 NTU
    - **Dissolved Oxygen:** ≥ 6 mg/L (expanded range: 0-20 mg/L)
    - **Conductivity:** < 1000 µS/cm (expanded range: 0-2000 µS/cm)
    - **Temperature:** < 30°C (expanded range: 0-50°C)
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;">
        <p>Developed with ❤️ using Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

# Main content area
tab1, tab2 = st.tabs(["🔍 Water Safety Check", "📊 Parameter Ranges"])

with tab1:
    # Input form with expanded ranges
    with st.form("water_quality_form"):
        st.subheader("Enter Water Quality Parameters")
        
        cols = st.columns(2)
        
        with cols[0]:
            st.markdown("""
            <div style="background-color:#e6f7ff;padding:15px;border-radius:10px;margin-bottom:20px;">
                <h4 style="color:#005b96;">Basic Parameters</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Expanded ranges
            ph = st.slider('pH Level (0-14 scale)', 0.0, 14.0, 7.0, 0.1, 
                          help="Measure of how acidic/basic water is (0-14 scale)")
            temperature = st.slider('Temperature (°C)', 0.0, 50.0, 22.0, 0.5,
                                  help="Water temperature measurement (0-50°C range)")
        
        with cols[1]:
            st.markdown("""
            <div style="background-color:#e6f7ff;padding:15px;border-radius:10px;margin-bottom:20px;">
                <h4 style="color:#005b96;">Quality Indicators</h4>
            </div>
            """, unsafe_allow_html=True)
            
            turbidity = st.slider('Turbidity (NTU)', 0.0, 3.0, 0.01, 0.01,
                                help="Measure of water clarity (0-100 NTU range)")
            dissolved_oxygen = st.slider('Dissolved Oxygen (mg/L)', 0.0, 200.0, 8.0, 0.1,
                                        help="Oxygen available for aquatic organisms (0-20 mg/L range)")
            conductivity = st.slider('Conductivity (µS/cm)', 0, 20000, 350, 10,
                                   help="Measure of water's ability to conduct electricity (0-2000 µS/cm range)")
        
        submitted = st.form_submit_button("Analyze Water Safety", 
                                         use_container_width=True,
                                         type="primary")

    # When form is submitted
    if submitted:
        # Create input dataframe
        input_data = pd.DataFrame([[ph, temperature, turbidity, dissolved_oxygen, conductivity]],
                                columns=['pH', 'Temperature (°C)', 'Turbidity (NTU)', 
                                        'Dissolved Oxygen (mg/L)', 'Conductivity (µS/cm)'])
        
        # Make prediction
        prediction = pipeline.predict(input_data)[0]
        prediction_proba = pipeline.predict_proba(input_data)[0]
        confidence = max(prediction_proba) * 100
        
        # Display results
        if prediction:
            st.markdown(f"""
            <div style="background-color:#e6f7e6;padding:20px;border-radius:10px;border-left:6px solid #4CAF50;margin:20px 0;">
                <h3 style="color:#4CAF50;margin-top:0;">✅ Safe for Consumption</h3>
                <p>This water meets WHO safety standards for human consumption.</p>
                <div style="background-color:#ffffff;padding:10px;border-radius:5px;">
                    <p style="margin:0;"><b>Confidence:</b> {confidence:.1f}%</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color:#ffebee;padding:20px;border-radius:10px;border-left:6px solid #f44336;margin:20px 0;">
                <h3 style="color:#f44336;margin-top:0;">❌ Not Safe for Consumption</h3>
                <p>This water does not meet WHO safety standards.</p>
                <div style="background-color:#ffffff;padding:10px;border-radius:5px;">
                    <p style="margin:0;"><b>Confidence:</b> {confidence:.1f}%</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Show detailed parameter analysis
        with st.expander("📈 Detailed Parameter Analysis", expanded=True):
            st.subheader("Parameter Evaluation")
            
            # Create evaluation metrics
            param_eval = {
                'Parameter': ['pH', 'Temperature', 'Turbidity', 'Dissolved Oxygen', 'Conductivity'],
                'Your Value': [ph, f"{temperature}°C", f"{turbidity} NTU", 
                               f"{dissolved_oxygen} mg/L", f"{conductivity} µS/cm"],
                'Safe Range': ['6.5-8.5', '< 30°C', '< 5 NTU', '≥ 6 mg/L', '< 1000 µS/cm'],
                'Status': [
                    '✅ Optimal' if 6.5 <= ph <= 8.5 else 
                    '⚠️ Alkaline' if ph > 8.5 else 
                    '⚠️ Acidic' if ph >= 4.5 else 
                    '❌ Dangerous',
                    
                    '✅ Normal' if temperature < 30 else 
                    '⚠️ Elevated' if temperature < 40 else 
                    '❌ Extreme',
                    
                    '✅ Clear' if turbidity < 5 else 
                    '⚠️ Cloudy' if turbidity < 20 else 
                    '❌ Very Turbid',
                    
                    '✅ Healthy' if dissolved_oxygen >= 6 else 
                    '⚠️ Low' if dissolved_oxygen >= 3 else 
                    '❌ Hypoxic',
                    
                    '✅ Normal' if conductivity < 1000 else 
                    '⚠️ High' if conductivity < 1500 else 
                    '❌ Very High'
                ],
                'Health Impact': [
                    'Ideal for drinking' if 6.5 <= ph <= 8.5 else 
                    'May cause irritation' if 8.5 < ph <= 9.5 or 5.5 <= ph < 6.5 else 
                    'Harmful to health',
                    
                    'No direct impact' if temperature < 30 else 
                    'Affects taste' if temperature < 40 else 
                    'Risk of scalding',
                    
                    'Clear and safe' if turbidity < 5 else 
                    'May harbor pathogens' if turbidity < 20 else 
                    'High pathogen risk',
                    
                    'Supports aquatic life' if dissolved_oxygen >= 6 else 
                    'Stressful for organisms' if dissolved_oxygen >= 3 else 
                    'Dangerously low oxygen',
                    
                    'Normal mineral content' if conductivity < 1000 else 
                    'Elevated minerals' if conductivity < 1500 else 
                    'Very high contamination risk'
                ]
            }
            
            # Display as dataframe with colored status
            eval_df = pd.DataFrame(param_eval)
            st.dataframe(
                eval_df,
                column_config={
                    "Status": st.column_config.TextColumn(
                        "Status",
                        help="Safety status of each parameter",
                        width="medium"
                    ),
                    "Health Impact": st.column_config.TextColumn(
                        "Health Impact",
                        help="Potential health implications",
                        width="large"
                    )
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Visual indicators
            st.markdown("### Safety Indicators")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("pH Level", f"{ph}", 
                          delta="Optimal" if 6.5 <= ph <= 8.5 else "Caution",
                          delta_color="normal" if 6.5 <= ph <= 8.5 else "off")
            
            with col2:
                st.metric("Dissolved Oxygen", f"{dissolved_oxygen} mg/L", 
                          delta="Healthy" if dissolved_oxygen >= 6 else "Low",
                          delta_color="normal" if dissolved_oxygen >= 6 else "off")
            
            with col3:
                st.metric("Conductivity", f"{conductivity} µS/cm", 
                          delta="Normal" if conductivity < 1000 else "High",
                          delta_color="normal" if conductivity < 1000 else "off")

with tab2:
    st.header("Extended Parameter Ranges")
    st.markdown("""
    This enhanced version includes wider measurement ranges for comprehensive water analysis:
    """)
    
    # Parameter range cards
    cols = st.columns(2)
    
    with cols[0]:
        st.markdown("""
        <div style="background-color:#e6f7ff;padding:15px;border-radius:10px;margin-bottom:20px;">
            <h4 style="color:#005b96;">pH Level</h4>
            <p><b>Range:</b> 0-14 (full pH scale)</p>
            <p><b>Safe Range:</b> 6.5-8.5</p>
            <p>Measures acidity/alkalinity. Extreme values can be harmful.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background-color:#e6f7ff;padding:15px;border-radius:10px;margin-bottom:20px;">
            <h4 style="color:#005b96;">Temperature</h4>
            <p><b>Range:</b> 0-50°C</p>
            <p><b>Safe Range:</b> < 30°C</p>
            <p>Affects chemical reactions and microbial growth.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown("""
        <div style="background-color:#e6f7ff;padding:15px;border-radius:10px;margin-bottom:20px;">
            <h4 style="color:#005b96;">Dissolved Oxygen</h4>
            <p><b>Range:</b> 0-20 mg/L</p>
            <p><b>Safe Range:</b> ≥ 6 mg/L</p>
            <p>Essential for aquatic life and water quality.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background-color:#e6f7ff;padding:15px;border-radius:10px;margin-bottom:20px;">
            <h4 style="color:#005b96;">Conductivity</h4>
            <p><b>Range:</b> 0-2000 µS/cm</p>
            <p><b>Safe Range:</b> < 1000 µS/cm</p>
            <p>Indicates dissolved inorganic salts and minerals.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Range visualization
    st.markdown("---")
    st.subheader("Parameter Range Visualization")
    
    # Create a dummy plot (replace with actual visualizations if desired)
    ranges = pd.DataFrame({
        'Parameter': ['pH', 'Temperature', 'Turbidity', 'Dissolved Oxygen', 'Conductivity'],
        'Min': [0, 0, 0, 0, 0],
        'Max': [14, 50, 100, 20, 2000],
        'Safe Min': [6.5, None, None, 6, None],
        'Safe Max': [8.5, 30, 5, None, 1000]
    })
    
    st.bar_chart(ranges.set_index('Parameter')[['Min', 'Max']])
    st.caption("Full measurement ranges for each parameter")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#666666;font-size:14px;">
    <p>AquaSafe Pro Water Quality Classifier v2.0</p>
    <p>For comprehensive water quality analysis</p>
</div>
""", unsafe_allow_html=True)