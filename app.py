import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import pickle
from tensorflow import keras

# Page configuration
st.set_page_config(
    page_title="Telco Churn Prediction", 
    layout="wide", 
    page_icon="🤖"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🤖 Telco Customer Churn Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Using Pre-trained ANN Model with SMOTEENN</p>', unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Load pre-trained model
@st.cache_resource
def load_pretrained_model():
    """Load the pre-trained model and scaler"""
    try:
        # Load your model file
        model = pickle.load(open('highest_accuracy_ann_model.pkl', 'rb'))
        
        # Create a default scaler (you should save and load your actual scaler)
        scaler = MinMaxScaler()
        
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Sidebar
st.sidebar.header("📦 Model Status")

# Try to load model automatically
if not st.session_state.model_loaded:
    try:
        model, scaler = load_pretrained_model()
        if model is not None:
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.model_loaded = True
            st.sidebar.success("✅ Model loaded successfully!")
            st.sidebar.info("Model: highest_accuracy_ann_model.pkl")
        else:
            st.sidebar.warning("⚠️ Model file not found")
            st.sidebar.info("Please upload your model files")
    except:
        st.sidebar.warning("⚠️ Please upload model files")

# File uploaders for model and scaler
st.sidebar.markdown("---")
st.sidebar.subheader("📁 Upload Model Files")

uploaded_model = st.sidebar.file_uploader("Upload Model (.pkl or .h5)", type=['pkl', 'h5'])
uploaded_scaler = st.sidebar.file_uploader("Upload Scaler (.pkl)", type=['pkl'])

if uploaded_model is not None:
    if st.sidebar.button("Load Uploaded Model"):
        try:
            # Load model
            if uploaded_model.name.endswith('.pkl'):
                st.session_state.model = pickle.load(uploaded_model)
            else:
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp:
                    tmp.write(uploaded_model.getbuffer())
                    st.session_state.model = keras.models.load_model(tmp.name)
            
            # Load scaler if provided, otherwise create default
            if uploaded_scaler is not None:
                st.session_state.scaler = pickle.load(uploaded_scaler)
            else:
                st.session_state.scaler = MinMaxScaler()
                st.sidebar.warning("No scaler uploaded, using default MinMaxScaler")
            
            st.session_state.model_loaded = True
            st.sidebar.success("✅ Model loaded!")
            st.rerun()
            
        except Exception as e:
            st.sidebar.error(f"Error loading: {str(e)}")

# Main content
if not st.session_state.model_loaded:
    st.info("👈 Please load the model using the sidebar")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📋 Instructions")
        st.markdown("""
        1. Upload your `highest_accuracy_ann_model.pkl` file
        2. Upload your scaler file (if available)
        3. Enter customer information
        4. Get instant churn prediction
        """)
    
    with col2:
        st.subheader("📊 Model Features")
        st.markdown("""
        - **Model**: Artificial Neural Network
        - **Training**: SMOTEENN Resampling
        - **Features**: 26 input features
        - **Output**: Churn probability (0-1)
        """)

else:
    st.success("🎉 Model is ready for predictions!")
    
    st.markdown("---")
    st.header("📝 Enter Customer Information")
    
    # Create input form
    st.markdown("### 📊 Numerical Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12, 
                                help="Number of months the customer has stayed with the company")
    with col2:
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0, 
                                         help="Amount charged to the customer monthly")
    with col3:
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=600.0,
                                       help="Total amount charged to the customer")
    
    st.markdown("---")
    st.markdown("### 🏷️ Service Features")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**Internet Service**")
        internet_service = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"], key='internet')
        online_security = st.selectbox("Online Security", ["No", "Yes"], key='security')
        online_backup = st.selectbox("Online Backup", ["No", "Yes"], key='backup')
    
    with col2:
        st.markdown("**Phone Service**")
        phone_service = st.selectbox("Phone Service", ["No", "Yes"], key='phone')
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"], key='lines')
        
    with col3:
        st.markdown("**Additional Services**")
        device_protection = st.selectbox("Device Protection", ["No", "Yes"], key='device')
        tech_support = st.selectbox("Tech Support", ["No", "Yes"], key='tech')
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"], key='tv')
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"], key='movies')
 
    with col4:
        st.markdown("**Contract & Payment**")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key='contract')
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"], key='paperless')
        payment_method = st.selectbox("Payment Method", 
                                     ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
                                     key='payment')
    
    st.markdown("---")
    st.markdown("### 👤 Customer Demographics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"], key='gender')
    with col2:
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"], key='senior')
    with col3:
        partner = st.selectbox("Has Partner", ["No", "Yes"], key='partner')
    with col4:
        dependents = st.selectbox("Has Dependents", ["No", "Yes"], key='dependents')
    
    st.markdown("---")
    
    # Predict button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True)
    
    if predict_button:
        # Prepare input data (this is a simplified example - adjust based on your actual feature encoding)
        # You need to encode the features exactly as they were during training
        
        # For demonstration, creating a sample input with 26 features
        # You should replace this with your actual feature engineering logic
        input_features = np.array([[
            tenure,
            monthly_charges,
            total_charges,
            1 if gender == "Male" else 0,
            1 if senior_citizen == "Yes" else 0,
            1 if partner == "Yes" else 0,
            1 if dependents == "Yes" else 0,
            1 if phone_service == "Yes" else 0,
            1 if multiple_lines == "Yes" else 0,
            1 if internet_service == "DSL" else 0,
            1 if internet_service == "Fiber optic" else 0,
            1 if internet_service == "No" else 0,
            1 if online_security == "Yes" else 0,
            1 if online_backup == "Yes" else 0,
            1 if device_protection == "Yes" else 0,
            1 if tech_support == "Yes" else 0,
            1 if streaming_tv == "Yes" else 0,
            1 if streaming_movies == "Yes" else 0,
            1 if contract == "One year"  else 0,
            1 if contract == "Two year" else 0,
            1 if contract == "Month-to-month" else 0,
            1 if paperless_billing == "Yes" else 0,
            1 if payment_method == "Electronic check" else 0,
            1 if payment_method == "Mailed check" else 0,
            1 if payment_method == "Bank transfer" else 0,
            1 if payment_method == "Credit card" else 0
        ]])
        

        try:
            # Scale the numerical features (first 3 columns)
            input_scaled = input_features.copy()
            # Note: You should use the actual fitted scaler from training
            # This is just for demonstration
            
            # Make prediction
            model = st.session_state.model
            prediction_proba = model.predict(input_scaled, verbose=0)[0][0]
            prediction = 1 if prediction_proba > 0.5 else 0
            
            st.markdown("---")
            st.header("🎯 Prediction Results")
            
            # Display result
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if prediction == 1:
                    st.error("### ⚠️ HIGH RISK OF CHURN")
                    st.metric("Churn Probability", f"{prediction_proba*100:.2f}%", 
                             delta=f"+{(prediction_proba-0.5)*100:.1f}%")
                else:
                    st.success("### ✅ LOW RISK OF CHURN")
                    st.metric("Retention Probability", f"{(1-prediction_proba)*100:.2f}%",
                             delta=f"+{(0.5-prediction_proba)*100:.1f}%")
            
            # Probability visualization
            st.subheader("📊 Churn Probability Gauge")
            fig, ax = plt.subplots(figsize=(10, 2))
            
            color = '#2ecc71' if prediction_proba < 0.5 else '#e74c3c'
            ax.barh([0], [prediction_proba], color=color, height=0.5, alpha=0.7)
            ax.barh([0], [1-prediction_proba], left=prediction_proba, color='lightgray', height=0.5, alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.5, 0.5)
            ax.set_yticks([])
            ax.set_xlabel('Probability', fontsize=12)
            ax.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold (0.5)')
            ax.set_title(f'Churn Probability: {prediction_proba:.4f}', fontsize=14, fontweight='bold')
            
            # Add percentage labels
            ax.text(prediction_proba/2, 0, f'{prediction_proba*100:.1f}%', 
                   ha='center', va='center', fontweight='bold', fontsize=12)
            ax.text(prediction_proba + (1-prediction_proba)/2, 0, f'{(1-prediction_proba)*100:.1f}%', 
                   ha='center', va='center', fontsize=10, color='gray')
            
            ax.legend(loc='upper right')
            st.pyplot(fig)
            
            # Customer details summary
            with st.expander("📋 View Customer Details"):
                details_df = pd.DataFrame({
                    'Feature': ['Tenure', 'Monthly Charges', 'Total Charges', 'Contract Type', 
                               'Internet Service', 'Payment Method', 'Senior Citizen', 'Has Partner'],
                    'Value': [f'{tenure} months', f'${monthly_charges:.2f}', f'${total_charges:.2f}', 
                             contract, internet_service, payment_method, senior_citizen, partner]
                })
                st.dataframe(details_df, use_container_width=True, hide_index=True)
            
            # Recommendations
            st.markdown("---")
            st.subheader("💡 Recommended Actions")
            
            if prediction == 1:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    **Immediate Actions:**
                    - 🎁 Offer 15-20% retention discount
                    - 📞 Schedule priority customer service call
                    - 💳 Review pricing and suggest better plan
                    - 📧 Send personalized retention email
                    """)
                with col2:
                    st.markdown("""
                    **Long-term Strategy:**
                    - 🎯 Enroll in loyalty program
                    - 🌟 Provide free service upgrades
                    - 📊 Monitor satisfaction closely
                    - 🤝 Assign dedicated account manager
                    """)
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    **Engagement Actions:**
                    - 😊 Customer likely to stay
                    - 🌟 Consider upselling premium services
                    - 📈 Continue regular satisfaction surveys
                    - 🎉 Offer loyalty rewards
                    """)
                with col2:
                    st.markdown("""
                    **Growth Opportunities:**
                    - 📱 Introduce new product features
                    - 🔄 Cross-sell additional services
                    - 💬 Request referrals
                    - ⭐ Feature as success story
                    """)
            
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            st.info("Make sure the model expects 26 features with proper scaling")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("🤖 Powered by TensorFlow & Streamlit")
st.sidebar.caption("Pre-trained ANN Model v1.0")