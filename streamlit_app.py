import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from quality_predictor import QualityPredictor, load_and_preprocess_data

# Page configuration
st.set_page_config(
    page_title="F&B Quality Prediction Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
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
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .quality-excellent {
        color: #28a745;
        font-weight: bold;
    }
    .quality-good {
        color: #17a2b8;
        font-weight: bold;
    }
    .quality-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .quality-poor {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the dataset"""
    try:
        # Try multiple paths for deployment compatibility
        data_paths = ["original_data.xlsx", "../original_data.xlsx"]
        for path in data_paths:
            try:
                df = load_and_preprocess_data(path)
                return df
            except FileNotFoundError:
                continue
        raise FileNotFoundError("Could not find original_data.xlsx in any expected location")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def load_model():
    """Load and cache the trained model"""
    predictor = QualityPredictor()
    try:
        # Try to load existing model
        predictor.load_model("fnb_quality_model.pkl")
    except:
        # If no model exists, train a new one
        df = load_data()
        if df is not None:
            with st.spinner("Training model for the first time..."):
                metrics, _, _, _ = predictor.train_model(df)
                predictor.save_model("fnb_quality_model.pkl")
    return predictor

def get_quality_status(score):
    """Get quality status and color based on score"""
    if score >= 85:
        return "Excellent", "quality-excellent", "#28a745"
    elif score >= 70:
        return "Good", "quality-good", "#17a2b8"
    elif score >= 50:
        return "Warning", "quality-warning", "#ffc107"
    else:
        return "Poor", "quality-poor", "#dc3545"

def create_gauge_chart(value, title, min_val=0, max_val=100):
    """Create a gauge chart for quality metrics"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': 80},
        gauge = {
            'axis': {'range': [None, max_val]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))
    
    fig.update_layout(height=300)
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🏭 F&B Process Quality Prediction Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Load data and model
    df = load_data()
    predictor = load_model()
    
    if df is None:
        st.error("Failed to load data. Please check if the data file exists.")
        return
    
    # Sidebar
    st.sidebar.title("🎛️ Control Panel")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Select Page",
        ["Real-time Prediction", "Batch Analysis", "Historical Data", "Model Performance"]
    )
    
    if page == "Real-time Prediction":
        show_realtime_prediction(predictor)
    elif page == "Batch Analysis":
        show_batch_analysis(df, predictor)
    elif page == "Historical Data":
        show_historical_data(df)
    elif page == "Model Performance":
        show_model_performance(df, predictor)

def show_realtime_prediction(predictor):
    """Real-time quality prediction interface with sliders"""
    st.header("🔄 Real-time Quality Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Input Process Parameters")
        
        # Product type selector
        st.markdown("**Product Type Selection**")
        product_type = st.selectbox(
            "Choose your product type (auto-detection will override this)",
            options=['Auto-detect', 'Baked Goods', 'Confectionery', 'Beverages', 'Dairy Products'],
            help="The system will auto-detect based on your ingredients, but you can pre-select for guidance"
        )
        
        st.markdown("---")
        
        # Create input forms with sliders
        input_col1, input_col2 = st.columns(2)
        
        with input_col1:
            st.markdown("**Ingredients (Use sliders for large-scale production)**")
            
            # Flour slider
            flour = st.slider(
                "Flour (kg)", 
                min_value=0.0, 
                max_value=10000.0, 
                value=10.0, 
                step=0.1,
                help="Primary ingredient for baked goods. Set to 0 for beverages/confectionery."
            )
            
            # Sugar slider
            sugar = st.slider(
                "Sugar (kg)", 
                min_value=0.0, 
                max_value=10000.0, 
                value=5.0, 
                step=0.1,
                help="Main ingredient for confectionery, sweetener for others."
            )
            
            # Yeast slider
            yeast = st.slider(
                "Yeast (kg)", 
                min_value=0.0, 
                max_value=1000.0, 
                value=2.0, 
                step=0.1,
                help="Required for fermentation in baked goods and some dairy. Set to 0 for confectionery/beverages."
            )
            
            # Salt slider
            salt = st.slider(
                "Salt (kg)", 
                min_value=0.0, 
                max_value=100.0, 
                value=1.0, 
                step=0.1,
                help="Flavor enhancer. Use sparingly - too much ruins the product!"
            )
            
            st.markdown("**Equipment Parameters**")
            mixer_speed = st.slider(
                "Mixer Speed (RPM)", 
                min_value=0, 
                max_value=500, 
                value=150, 
                step=5,
                help="Lower speeds for larger batches, higher for smaller ones."
            )
        
        with input_col2:
            st.markdown("**Temperature Control**")
            
            water_temp = st.slider(
                "Water Temperature (°C)", 
                min_value=0, 
                max_value=100, 
                value=25, 
                step=1,
                help="Initial water temperature for mixing."
            )
            
            mixing_temp = st.slider(
                "Mixing Temperature (°C)", 
                min_value=10, 
                max_value=80, 
                value=35, 
                step=1,
                help="Temperature during mixing process."
            )
            
            fermentation_temp = st.slider(
                "Fermentation Temperature (°C)", 
                min_value=15, 
                max_value=60, 
                value=35, 
                step=1,
                help="Temperature for yeast activation and fermentation."
            )
            
            oven_temp = st.slider(
                "Oven Temperature (°C)", 
                min_value=0, 
                max_value=500, 
                value=180, 
                step=5,
                help="Baking temperature. Set to 0 for non-baked products. WARNING: >300°C is dangerous!"
            )
            
            oven_humidity = st.slider(
                "Oven Humidity (%)", 
                min_value=10, 
                max_value=90, 
                value=45, 
                step=1,
                help="Humidity level during baking."
            )
        
        # Real-time ingredient ratio display
        st.markdown("---")
        st.markdown("**Real-time Ingredient Analysis**")
        
        total_mass = flour + sugar + yeast + salt
        ratio_col1, ratio_col2, ratio_col3, ratio_col4 = st.columns(4)
        
        with ratio_col1:
            if total_mass > 0:
                flour_pct = (flour / total_mass) * 100
                st.metric("Flour %", f"{flour_pct:.1f}%")
        
        with ratio_col2:
            if total_mass > 0:
                sugar_pct = (sugar / total_mass) * 100
                st.metric("Sugar %", f"{sugar_pct:.1f}%")
        
        with ratio_col3:
            if total_mass > 0:
                yeast_pct = (yeast / total_mass) * 100
                st.metric("Yeast %", f"{yeast_pct:.1f}%")
        
        with ratio_col4:
            if total_mass > 0:
                salt_pct = (salt / total_mass) * 100
                st.metric("Salt %", f"{salt_pct:.1f}%")
        
        # Scale indicator
        if total_mass > 1000:
            st.info(f"🏭 **Industrial Scale Production**: {total_mass:.0f}kg total batch")
        elif total_mass > 100:
            st.info(f"🏢 **Commercial Scale Production**: {total_mass:.0f}kg total batch")
        else:
            st.info(f"🏠 **Small Scale Production**: {total_mass:.0f}kg total batch")
        
        # Prediction button
        if st.button("🔮 Predict Quality", type="primary"):
            # Create input dataframe
            input_data = pd.DataFrame({
                'Flour (kg)': [flour],
                'Sugar (kg)': [sugar],
                'Yeast (kg)': [yeast],
                'Salt (kg)': [salt],
                'Mixer Speed (RPM)': [mixer_speed],
                'Water Temp (C)': [water_temp],
                'Mixing Temp (C)': [mixing_temp],
                'Fermentation Temp (C)': [fermentation_temp],
                'Oven Temp (C)': [oven_temp],
                'Oven Humidity (%)': [oven_humidity]
            })
            
            # Make prediction
            try:
                results = predictor.predict_quality(input_data)
                
                predicted_quality = results['predicted_quality'][0]
                realistic_quality = results['realistic_quality'].iloc[0]
                product_types = results.get('product_types', ['unknown'])
                detected_product = product_types[0] if product_types else 'unknown'
                is_anomaly = results['is_anomaly'][0]
                anomaly_score = results['anomaly_scores'][0]
                
                # Display results in the right column
                with col2:
                    st.subheader("📊 Prediction Results")
                    
                    # Product type detection
                    product_names = {
                        'baked_goods': 'Baked Goods 🍞',
                        'confectionery': 'Confectionery 🍬',
                        'beverages': 'Beverages 🥤',
                        'dairy': 'Dairy Products 🧀',
                        'unknown': 'Unknown Product'
                    }
                    
                    st.info(f"**Detected Product:** {product_names.get(detected_product, detected_product)}")
                    
                    # Quality Score
                    status, css_class, color = get_quality_status(realistic_quality)
                    st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Quality Score", f"{realistic_quality:.1f}%", f"{realistic_quality - 80:.1f}%")
                    st.markdown(f'<p class="{css_class}">Status: {status}</p>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Anomaly Detection
                    if is_anomaly:
                        st.error("⚠️ Anomaly Detected!")
                        st.write(f"Anomaly Score: {anomaly_score:.3f}")
                    else:
                        st.success("✅ Normal Process")
                    
                    # Gauge chart
                    fig_gauge = create_gauge_chart(realistic_quality, "Quality Score")
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Generate and display comprehensive report
                st.markdown("---")
                st.subheader("� Comprehensive Quality Report")
                
                try:
                    quality_report = predictor.generate_quality_report(input_data, results)
                    st.markdown(quality_report)
                except Exception as e:
                    st.error(f"Error generating report: {e}")
                    
                    # Fallback basic analysis
                    st.markdown("### Basic Analysis")
                    ratios = predictor.calculate_ingredient_ratios(flour, sugar, yeast, salt)
                    
                    if flour == 0 and sugar > 0:
                        st.info("🍬 **Confectionery Product Detected** - Sugar-based product")
                    elif flour > 0 and yeast > 0:
                        st.info("🍞 **Baked Goods Detected** - Bread/cake product")
                    elif flour == 0 and sugar == 0:
                        st.info("🥤 **Beverage Product Detected** - Liquid product")
                    
                    # Critical warnings
                    if salt > max(flour, sugar) * 0.5:
                        st.error("🚨 **CRITICAL**: Salt content is dangerously high!")
                    
                    if oven_temp > 300:
                        st.error("🚨 **DANGER**: Oven temperature is extremely high!")
                        
            except Exception as e:
                st.error(f"Error making prediction: {e}")
                st.info("Please check your inputs and try again.")

def show_batch_analysis(df, predictor):
    """Batch analysis and comparison"""
    st.header("📦 Batch Analysis")
    
    # Select batch
    available_batches = sorted(df['Batch_ID'].unique())
    selected_batch = st.selectbox("Select Batch ID", available_batches)
    
    # Filter data for selected batch
    batch_data = df[df['Batch_ID'] == selected_batch].copy()
    
    if len(batch_data) == 0:
        st.warning("No data found for selected batch.")
        return
    
    # Calculate quality scores for the batch
    batch_data['Quality_Score'] = batch_data.apply(predictor.calculate_realistic_quality_score, axis=1)
    
    # Display batch summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Batch Size", f"{len(batch_data)} samples")
    
    with col2:
        avg_quality = batch_data['Quality_Score'].mean()
        st.metric("Average Quality", f"{avg_quality:.1f}%")
    
    with col3:
        min_quality = batch_data['Quality_Score'].min()
        st.metric("Minimum Quality", f"{min_quality:.1f}%")
    
    with col4:
        max_quality = batch_data['Quality_Score'].max()
        st.metric("Maximum Quality", f"{max_quality:.1f}%")
    
    # Time series plot
    st.subheader("📈 Quality Over Time")
    
    fig_time = px.line(batch_data, x='Time', y='Quality_Score', 
                       title=f'Quality Score Over Time - Batch {selected_batch}')
    fig_time.add_hline(y=80, line_dash="dash", line_color="green", 
                       annotation_text="Target Quality (80%)")
    fig_time.add_hline(y=50, line_dash="dash", line_color="red", 
                       annotation_text="Minimum Acceptable (50%)")
    st.plotly_chart(fig_time, use_container_width=True)
    
    # Parameter correlation with quality
    st.subheader("🔗 Parameter Correlation with Quality")
    
    # Calculate correlations
    numeric_cols = batch_data.select_dtypes(include=[np.number]).columns
    correlations = batch_data[numeric_cols].corr()['Quality_Score'].abs().sort_values(ascending=False)
    correlations = correlations.drop('Quality_Score')  # Remove self-correlation
    
    fig_corr = px.bar(x=correlations.values, y=correlations.index, orientation='h',
                      title='Parameter Correlation with Quality Score')
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Parameter distribution
    st.subheader("📊 Parameter Distributions")
    
    # Select parameters to show
    param_cols = st.multiselect(
        "Select parameters to visualize",
        options=['Flour (kg)', 'Sugar (kg)', 'Yeast (kg)', 'Salt (kg)', 
                'Oven Temp (C)', 'Oven Humidity (%)', 'Mixer Speed (RPM)'],
        default=['Flour (kg)', 'Sugar (kg)', 'Oven Temp (C)']
    )
    
    if param_cols:
        fig_dist = make_subplots(rows=1, cols=len(param_cols), 
                                subplot_titles=param_cols)
        
        for i, col in enumerate(param_cols):
            fig_dist.add_trace(
                go.Histogram(x=batch_data[col], name=col, showlegend=False),
                row=1, col=i+1
            )
        
        fig_dist.update_layout(title="Parameter Distributions", height=400)
        st.plotly_chart(fig_dist, use_container_width=True)

def show_historical_data(df):
    """Historical data analysis"""
    st.header("📈 Historical Data Analysis")
    
    # Calculate quality scores for all data
    predictor = QualityPredictor()
    df_analysis = df.copy()
    df_analysis['Quality_Score'] = df_analysis.apply(predictor.calculate_realistic_quality_score, axis=1)
    
    # Overall statistics
    st.subheader("📊 Overall Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Batches", df['Batch_ID'].nunique())
    
    with col2:
        st.metric("Total Samples", len(df))
    
    with col3:
        avg_quality_all = df_analysis['Quality_Score'].mean()
        st.metric("Average Quality", f"{avg_quality_all:.1f}%")
    
    with col4:
        good_quality_pct = (df_analysis['Quality_Score'] >= 70).mean() * 100
        st.metric("Good Quality %", f"{good_quality_pct:.1f}%")
    
    # Quality distribution
    st.subheader("🎯 Quality Score Distribution")
    
    fig_dist = px.histogram(df_analysis, x='Quality_Score', nbins=20,
                           title='Distribution of Quality Scores')
    fig_dist.add_vline(x=70, line_dash="dash", line_color="green", 
                       annotation_text="Good Quality Threshold")
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Batch comparison
    st.subheader("🔄 Batch Comparison")
    
    batch_stats = df_analysis.groupby('Batch_ID')['Quality_Score'].agg(['mean', 'std', 'min', 'max']).reset_index()
    batch_stats.columns = ['Batch_ID', 'Avg_Quality', 'Quality_Std', 'Min_Quality', 'Max_Quality']
    
    fig_batch = px.box(df_analysis, x='Batch_ID', y='Quality_Score',
                       title='Quality Score Distribution by Batch')
    st.plotly_chart(fig_batch, use_container_width=True)
    
    # Parameter trends
    st.subheader("📉 Parameter Trends")
    
    # Select parameter to analyze
    param_to_analyze = st.selectbox(
        "Select parameter to analyze trends",
        options=['Flour (kg)', 'Sugar (kg)', 'Yeast (kg)', 'Salt (kg)', 
                'Oven Temp (C)', 'Oven Humidity (%)', 'Mixer Speed (RPM)']
    )
    
    # Calculate batch averages
    batch_trends = df_analysis.groupby('Batch_ID')[param_to_analyze].mean().reset_index()
    
    fig_trend = px.line(batch_trends, x='Batch_ID', y=param_to_analyze,
                       title=f'{param_to_analyze} Trend Across Batches')
    
    # Add ideal line
    ideal_value = predictor.base_ideal_conditions.get(param_to_analyze, 0)
    if ideal_value > 0:
        fig_trend.add_hline(y=ideal_value, line_dash="dash", line_color="red",
                           annotation_text=f"Ideal: {ideal_value}")
    
    st.plotly_chart(fig_trend, use_container_width=True)

def show_model_performance(df, predictor):
    """Model performance and feature importance"""
    st.header("🧠 Model Performance")
    
    # Retrain model to get metrics
    with st.spinner("Analyzing model performance..."):
        metrics, X_test, y_test, y_pred = predictor.train_model(df)
    
    # Performance metrics
    st.subheader("📏 Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        test_r2 = metrics['r2']
        st.metric("Test R² Score", f"{test_r2:.4f}")
    
    with col2:
        cv_mean = metrics.get('cv_mean', 0)
        cv_std = metrics.get('cv_std', 0)
        st.metric("Cross-Validation R²", f"{cv_mean:.4f} ± {cv_std:.4f}")
    
    with col3:
        overfitting_gap = metrics.get('overfitting_gap', 0)
        st.metric("Overfitting Gap", f"{overfitting_gap:.4f}")
        if overfitting_gap > 0.15:
            st.error("⚠️ Model shows signs of overfitting!")
        else:
            st.success("✅ Model generalization looks good")
    
    with col4:
        mae = metrics['mae']
        st.metric("Mean Absolute Error", f"{mae:.2f}")
    
    # Additional metrics
    col5, col6, col7 = st.columns(3)
    with col5:
        rmse = np.sqrt(metrics['mse'])
        st.metric("Root Mean Square Error", f"{rmse:.2f}")
    
    with col6:
        oob_score = metrics.get('oob_score', None)
        if oob_score:
            st.metric("Out-of-Bag Score", f"{oob_score:.4f}")
        else:
            st.metric("Out-of-Bag Score", "N/A")
            
    with col7:
        generalization = metrics.get('generalization_score', 0)
        st.metric("Generalization Score", f"{generalization:.4f}")
    
    # Prediction vs Actual plot
    st.subheader("🎯 Prediction vs Actual")
    
    comparison_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    })
    
    fig_scatter = px.scatter(comparison_df, x='Actual', y='Predicted',
                            title='Predicted vs Actual Quality Scores')
    fig_scatter.add_shape(type="line", x0=0, y0=0, x1=100, y1=100,
                         line=dict(dash="dash", color="red"))
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Feature importance
    st.subheader("🎯 Feature Importance")
    
    importance_df = pd.DataFrame({
        'Feature': list(metrics['feature_importance'].keys()),
        'Importance': list(metrics['feature_importance'].values())
    }).sort_values('Importance', ascending=False)
    
    fig_importance = px.bar(importance_df.head(10), x='Importance', y='Feature', 
                           orientation='h', title='Top 10 Most Important Features')
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Model interpretation
    st.subheader("🔍 Model Interpretation")
    
    # Overfitting analysis
    overfitting_gap = metrics.get('overfitting_gap', 0)
    is_overfitted = metrics.get('is_overfitted', False)
    
    if is_overfitted:
        st.warning("⚠️ **Overfitting Detected!**")
        st.write(f"""
        The model shows signs of overfitting with a gap of {overfitting_gap:.3f} between training and test performance.
        
        **What this means:**
        - Training R² = {metrics.get('train_r2', 0):.4f}
        - Test R² = {metrics['r2']:.4f}
        - Gap = {overfitting_gap:.4f} (> 0.1 indicates overfitting)
        
        **Mitigation steps taken:**
        - Added noise to simulate real-world variability
        - Reduced model complexity (max_depth=6, n_estimators=50)
        - Increased minimum samples per split/leaf
        - Used bootstrap sampling for regularization
        """)
    else:
        st.success("✅ **Good Model Generalization**")
        st.write(f"""
        The model shows good generalization with minimal overfitting.
        
        **Performance Analysis:**
        - Training R² = {metrics.get('train_r2', 0):.4f}
        - Test R² = {metrics['r2']:.4f}
        - Cross-Validation R² = {metrics.get('cv_mean', 0):.4f} ± {metrics.get('cv_std', 0):.4f}
        - Gap = {overfitting_gap:.4f} (< 0.15 indicates good generalization)
        
        **Cross-Validation Results:**
        The model was validated using 5-fold cross-validation to ensure robust performance estimates.
        """)
    
    st.write("""
    **Key Insights from the Model:**
    
    1. **Feature Importance**: The chart below shows which parameters have the most impact on quality prediction.
    
    2. **Model Accuracy**: The R² score indicates how well the model explains the variance in quality scores.
    
    3. **Prediction Reliability**: The scatter plot shows the relationship between predicted and actual values.
    
    4. **Quality Factors**: The model considers ingredient ratios, temperature controls, and equipment parameters.
    
    5. **Real-world Variability**: Noise has been added to simulate actual manufacturing conditions.
    """)
    
    # Model details
    with st.expander("🔧 Model Configuration Details"):
        st.write(f"""
        **Random Forest Parameters:**
        - Number of trees: 50 (reduced to prevent overfitting)
        - Maximum depth: 6 (limited to prevent overfitting)
        - Minimum samples per split: 10
        - Minimum samples per leaf: 5
        - Maximum features: sqrt(total_features)
        - Bootstrap sampling: Enabled
        - Out-of-bag scoring: Enabled
        
        **Data Preprocessing:**
        - Added 8% noise to simulate real-world variability
        - Stratified train/test split (70/30)
        - StandardScaler for feature normalization
        - 23 engineered features including ratios and deviations
        
        **Anomaly Detection:**
        - Isolation Forest with 10% contamination threshold
        - Trained on the same scaled features as the main model
        """)
    
    # Ideal vs Current comparison
    st.subheader("⚖️ Ideal vs Current Conditions")
    
    # Calculate current averages
    current_avg = df.mean()
    
    comparison_data = []
    for param, ideal_val in predictor.base_ideal_conditions.items():
        if param in current_avg:
            current_val = current_avg[param]
            deviation = ((current_val - ideal_val) / ideal_val) * 100
            comparison_data.append({
                'Parameter': param,
                'Ideal': ideal_val,
                'Current Average': current_val,
                'Deviation (%)': deviation
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)

if __name__ == "__main__":
    main()
