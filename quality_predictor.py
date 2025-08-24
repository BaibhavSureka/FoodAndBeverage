import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

class QualityPredictor:
    """
    F&B Process Quality Prediction Model
    
    This class implements a sophisticated quality prediction system for various F&B products
    including baked goods, confectionery, beverages, and other food processing scenarios.
    It handles large-scale production and different product types.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        
        # Product type definitions
        self.product_types = {
            'baked_goods': {
                'name': 'Baked Goods (Bread, Cakes, Pastries)',
                'requires_flour': True,
                'requires_yeast': True,
                'requires_oven': True,
                'ideal_ratios': {'sugar': 0.5, 'yeast': 0.2, 'salt': 0.1}
            },
            'confectionery': {
                'name': 'Confectionery (Candies, Sugar Syrups)',
                'requires_flour': False,
                'requires_yeast': False,
                'requires_oven': False,
                'ideal_ratios': {'sugar': 2.0, 'yeast': 0.0, 'salt': 0.05}
            },
            'beverages': {
                'name': 'Beverages (Juices, Soft Drinks)',
                'requires_flour': False,
                'requires_yeast': False,
                'requires_oven': False,
                'ideal_ratios': {'sugar': 0.3, 'yeast': 0.0, 'salt': 0.02}
            },
            'dairy': {
                'name': 'Dairy Products (Cheese, Yogurt)',
                'requires_flour': False,
                'requires_yeast': True,
                'requires_oven': False,
                'ideal_ratios': {'sugar': 0.1, 'yeast': 0.05, 'salt': 0.03}
            }
        }
        
        # Ideal conditions for reference (scalable)
        self.base_ideal_conditions = {
            'Flour (kg)': 10,
            'Sugar (kg)': 5,
            'Yeast (kg)': 2,
            'Salt (kg)': 1,
            'Mixer Speed (RPM)': 150,
            'Water Temp (C)': 25,
            'Mixing Temp (C)': 35,
            'Fermentation Temp (C)': 35,
            'Oven Temp (C)': 180,
            'Oven Humidity (%)': 45
        }
        
        # Acceptable ranges for quality parameters (percentage of ideal)
        self.acceptable_ranges = {
            'Flour (kg)': (0.8, 1.2),      # 80-120% of ideal
            'Sugar (kg)': (0.6, 1.4),      # 60-140% of ideal  
            'Yeast (kg)': (0.5, 2.0),      # 50-200% of ideal
            'Salt (kg)': (0.3, 3.0),       # 30-300% of ideal
            'Mixer Speed (RPM)': (0.7, 1.3),
            'Water Temp (C)': (0.8, 1.2),
            'Mixing Temp (C)': (0.8, 1.2),
            'Fermentation Temp (C)': (0.8, 1.2),
            'Oven Temp (C)': (0.8, 1.2),
            'Oven Humidity (%)': (0.7, 1.3)
        }
    
    def detect_product_type(self, flour, sugar, yeast, oven_temp):
        """Automatically detect the most likely product type based on ingredients"""
        total_ingredients = flour + sugar + yeast
        
        if total_ingredients == 0:
            return 'beverages'
        
        sugar_ratio = sugar / max(total_ingredients, 0.001)
        flour_ratio = flour / max(total_ingredients, 0.001)
        
        # Decision logic based on ratios
        if flour_ratio > 0.5 and yeast > 0 and oven_temp > 100:
            return 'baked_goods'
        elif sugar_ratio > 0.7 and flour_ratio < 0.1:
            return 'confectionery'
        elif flour_ratio < 0.1 and sugar_ratio < 0.5:
            return 'beverages'
        elif yeast > 0 and flour_ratio < 0.3:
            return 'dairy'
        else:
            return 'baked_goods'  # Default
    
    def get_ideal_ratios_for_product(self, product_type, base_ingredient_amount):
        """Get ideal ingredient ratios based on product type and scale"""
        if product_type not in self.product_types:
            product_type = 'baked_goods'
        
        ideal_ratios = self.product_types[product_type]['ideal_ratios']
        
        # Scale ratios based on base ingredient (flour for baked goods, sugar for confectionery, etc.)
        if product_type == 'baked_goods':
            base = base_ingredient_amount  # flour
        elif product_type in ['confectionery', 'beverages']:
            base = base_ingredient_amount  # sugar as base
        else:
            base = max(base_ingredient_amount, 1)  # avoid division by zero
        
        return {
            'sugar': ideal_ratios['sugar'] * base,
            'yeast': ideal_ratios['yeast'] * base,
            'salt': ideal_ratios['salt'] * base
        }
        
    def calculate_ingredient_ratios(self, flour, sugar, yeast, salt):
        """Calculate ingredient ratios relative to flour as base"""
        if flour <= 0:
            return {'sugar_ratio': 0, 'yeast_ratio': 0, 'salt_ratio': 0}
            
        return {
            'sugar_ratio': sugar / flour,
            'yeast_ratio': yeast / flour,
            'salt_ratio': salt / flour
        }
    
    def calculate_realistic_quality_score(self, row):
        """
        Calculate quality score based on realistic manufacturing principles
        Now supports multiple product types and large-scale production
        
        Quality factors:
        1. Ingredient ratios based on product type (40% weight)
        2. Process temperatures (30% weight)
        3. Equipment parameters (20% weight)
        4. Edge case penalties (10% weight)
        """
        
        # Extract values
        flour = row['Flour (kg)']
        sugar = row['Sugar (kg)']
        yeast = row['Yeast (kg)']
        salt = row['Salt (kg)']
        mixer_speed = row['Mixer Speed (RPM)']
        water_temp = row['Water Temp (C)']
        mixing_temp = row['Mixing Temp (C)']
        fermentation_temp = row['Fermentation Temp (C)']
        oven_temp = row['Oven Temp (C)']
        oven_humidity = row['Oven Humidity (%)']
        
        # Auto-detect product type
        product_type = self.detect_product_type(flour, sugar, yeast, oven_temp)
        product_info = self.product_types[product_type]
        
        # Calculate ingredient ratios
        ratios = self.calculate_ingredient_ratios(flour, sugar, yeast, salt)
        
        # Get ideal ratios for detected product type
        if product_type == 'baked_goods' and flour > 0:
            ideal_ratios = self.get_ideal_ratios_for_product(product_type, flour)
            base_ingredient = flour
        elif product_type in ['confectionery', 'beverages'] and sugar > 0:
            ideal_ratios = self.get_ideal_ratios_for_product(product_type, sugar)
            base_ingredient = sugar
        else:
            # Fallback to original ratios
            ideal_ratios = {'sugar': 5, 'yeast': 2, 'salt': 1}
            base_ingredient = max(flour, sugar, 1)
        
        quality_score = 100.0  # Start with perfect score
        
        # 1. INGREDIENT RATIO QUALITY (40% weight)
        ratio_quality = 100.0
        
        # Product-specific quality assessment
        if product_type == 'confectionery':
            # For confectionery, sugar is the main ingredient
            if flour > sugar * 0.1:  # Too much flour in candy/syrup
                ratio_quality -= 40
            if yeast > 0.01:  # Yeast shouldn't be in most confectionery
                ratio_quality -= 30
            # Salt ratio is critical for taste balance
            if salt > sugar * 0.2:  # Too much salt
                ratio_quality -= 50
                
        elif product_type == 'beverages':
            # For beverages, minimal solid ingredients
            if flour > 0.1:  # Flour shouldn't be in beverages
                ratio_quality -= 60
            if yeast > 0.01 and product_type != 'fermented_beverage':
                ratio_quality -= 40
            # Sugar content check
            if sugar > base_ingredient * 2:  # Too sweet
                ratio_quality -= 30
                
        elif product_type == 'dairy':
            # For dairy, balanced fermentation is key
            if flour > sugar:  # Usually no flour in dairy
                ratio_quality -= 50
            # Yeast/culture balance
            yeast_deviation = abs(ratios['yeast_ratio'] - 0.05) / 0.05
            if yeast_deviation > 2.0:
                ratio_quality -= 40
                
        else:  # baked_goods - original logic applies
            ideal_sugar_ratio = ideal_ratios['sugar'] / base_ingredient
            ideal_yeast_ratio = ideal_ratios['yeast'] / base_ingredient
            ideal_salt_ratio = ideal_ratios['salt'] / base_ingredient
            
            # Sugar ratio impact
            if base_ingredient > 0:
                sugar_deviation = abs(ratios['sugar_ratio'] - ideal_sugar_ratio) / max(ideal_sugar_ratio, 0.01)
                if sugar_deviation > 2.0:
                    ratio_quality -= 40
                elif sugar_deviation > 1.0:
                    ratio_quality -= 25
                elif sugar_deviation > 0.5:
                    ratio_quality -= 15
                
                # Yeast ratio impact (critical for rise)
                yeast_deviation = abs(ratios['yeast_ratio'] - ideal_yeast_ratio) / max(ideal_yeast_ratio, 0.01)
                if yeast_deviation > 3.0:
                    ratio_quality -= 35
                elif yeast_deviation > 1.5:
                    ratio_quality -= 20
                elif yeast_deviation > 0.8:
                    ratio_quality -= 10
                
                # Salt ratio impact
                salt_deviation = abs(ratios['salt_ratio'] - ideal_salt_ratio) / max(ideal_salt_ratio, 0.01)
                if ratios['salt_ratio'] > ideal_salt_ratio * 5:
                    ratio_quality -= 60
                elif ratios['salt_ratio'] > ideal_salt_ratio * 3:
                    ratio_quality -= 40
                elif salt_deviation > 2.0:
                    ratio_quality -= 25
                elif salt_deviation > 1.0:
                    ratio_quality -= 10
        
        # 2. TEMPERATURE QUALITY (30% weight)
        temp_quality = 100.0
        
        # Product-specific temperature requirements
        if product_info['requires_oven']:
            # Oven temperature critical for baked goods
            oven_temp_deviation = abs(oven_temp - 180) / 180
            if oven_temp > 300:
                temp_quality -= 60
            elif oven_temp > 250:
                temp_quality -= 40
            elif oven_temp_deviation > 0.3:
                temp_quality -= 25
            elif oven_temp_deviation > 0.15:
                temp_quality -= 10
        else:
            # Non-baked products shouldn't use high oven temperatures
            if oven_temp > 100:
                temp_quality -= 30  # Unnecessary heating
        
        # Fermentation temperature (important for yeast-based products)
        if product_info['requires_yeast'] and yeast > 0:
            ferm_temp_deviation = abs(fermentation_temp - 35) / 35
            if ferm_temp_deviation > 0.4:
                temp_quality -= 20
            elif ferm_temp_deviation > 0.2:
                temp_quality -= 10
        
        # Water and mixing temperature
        water_temp_deviation = abs(water_temp - 25) / 25
        mixing_temp_deviation = abs(mixing_temp - 35) / 35
        
        if water_temp_deviation > 0.5 or mixing_temp_deviation > 0.3:
            temp_quality -= 15
        
        # 3. EQUIPMENT PARAMETERS (20% weight)
        equipment_quality = 100.0
        
        # Scale-appropriate mixer speed
        # Larger batches may need different speeds
        total_mass = flour + sugar + yeast + salt
        if total_mass > 100:  # Large scale production
            ideal_mixer_speed = 120  # Slower for large batches
        else:
            ideal_mixer_speed = 150
        
        mixer_deviation = abs(mixer_speed - ideal_mixer_speed) / ideal_mixer_speed
        if mixer_deviation > 0.5:
            equipment_quality -= 30
        elif mixer_deviation > 0.3:
            equipment_quality -= 15
        
        # Humidity control
        humidity_deviation = abs(oven_humidity - 45) / 45
        if humidity_deviation > 0.4:
            equipment_quality -= 20
        elif humidity_deviation > 0.2:
            equipment_quality -= 10
        
        # 4. EDGE CASE PENALTIES (10% weight)
        edge_penalty = 0
        
        # Extreme scenarios
        total_ingredients = flour + sugar + yeast + salt
        if total_ingredients <= 0:
            edge_penalty += 100  # No ingredients = no product
        
        # Product-specific edge cases
        if product_type == 'baked_goods' and flour <= 0:
            edge_penalty += 90  # No flour = no baked goods
        
        if product_type == 'confectionery' and sugar <= 0:
            edge_penalty += 90  # No sugar = no confectionery
        
        # Catastrophic ingredient ratios
        if salt > max(flour, sugar) * 2.0:  # Salt disaster
            edge_penalty += 85
        elif salt > max(flour, sugar) * 1.0:
            edge_penalty += 60
        elif salt > max(flour, sugar) * 0.5:
            edge_penalty += 30
        
        # Yeast extremes
        if yeast > max(flour, sugar) * 0.8:  # Too much yeast
            edge_penalty += 30
        elif product_info['requires_yeast'] and yeast == 0:
            edge_penalty += 40  # Missing required yeast
        
        # Temperature disasters
        if oven_temp > 800:  # Extreme temperature
            edge_penalty += 85
        elif oven_temp > 500:
            edge_penalty += 60
        elif oven_temp > 300:
            edge_penalty += 40
        elif product_info['requires_oven'] and oven_temp < 100:
            edge_penalty += 30
        
        # Scale-related issues
        if total_ingredients > 10000:  # Very large scale
            # Check if ratios are maintained at scale
            if abs(ratios['salt_ratio'] - 0.1) > 0.05:  # Salt ratio critical at scale
                edge_penalty += 20
        
        # Combine all quality factors with weights
        final_quality = (
            ratio_quality * 0.4 +
            temp_quality * 0.3 +
            equipment_quality * 0.2 +
            max(0, 100 - edge_penalty) * 0.1
        )
        
        # Store product type as instance variable for access
        self.last_detected_product_type = product_type
        
        # Ensure quality is between 0 and 100
        return max(0, min(100, final_quality))
    
    def generate_quality_report(self, input_data, results):
        """Generate a comprehensive NLP report about the quality prediction"""
        flour = input_data['Flour (kg)'].iloc[0]
        sugar = input_data['Sugar (kg)'].iloc[0]
        yeast = input_data['Yeast (kg)'].iloc[0]
        salt = input_data['Salt (kg)'].iloc[0]
        oven_temp = input_data['Oven Temp (C)'].iloc[0]
        
        quality_score = results['realistic_quality'].iloc[0]
        product_type = self.detect_product_type(flour, sugar, yeast, oven_temp)
        product_info = self.product_types[product_type]
        
        # Generate comprehensive report
        report = []
        
        # Header
        report.append(f"## Quality Assessment Report")
        report.append(f"**Detected Product Type:** {product_info['name']}")
        report.append(f"**Overall Quality Score:** {quality_score:.1f}%")
        
        # Quality status
        if quality_score >= 85:
            status = "EXCELLENT"
            emoji = "🟢"
        elif quality_score >= 70:
            status = "GOOD"
            emoji = "🟡"
        elif quality_score >= 50:
            status = "WARNING"
            emoji = "🟠"
        else:
            status = "POOR"
            emoji = "🔴"
        
        report.append(f"**Quality Status:** {emoji} {status}")
        report.append("")
        
        # Production scale analysis
        total_mass = flour + sugar + yeast + salt
        if total_mass > 1000:
            scale = "Industrial Scale"
            scale_desc = f"This is a large-scale production batch with {total_mass:.0f}kg total ingredients."
        elif total_mass > 100:
            scale = "Commercial Scale"
            scale_desc = f"This is a medium-scale commercial batch with {total_mass:.0f}kg total ingredients."
        else:
            scale = "Small Scale"
            scale_desc = f"This is a small-scale batch with {total_mass:.0f}kg total ingredients."
        
        report.append(f"### Production Scale Analysis")
        report.append(f"**Scale:** {scale}")
        report.append(scale_desc)
        report.append("")
        
        # Ingredient analysis
        report.append(f"### Ingredient Analysis")
        ratios = self.calculate_ingredient_ratios(flour, sugar, yeast, salt)
        
        # Product-specific analysis
        if product_type == 'baked_goods':
            report.append("**Analysis for Baked Goods:**")
            if flour == 0:
                report.append("⚠️ **CRITICAL:** No flour detected - cannot produce baked goods!")
            else:
                report.append(f"- Flour: {flour:.1f}kg (Base ingredient)")
                report.append(f"- Sugar to Flour Ratio: {ratios['sugar_ratio']:.2f} (Ideal: ~0.50)")
                report.append(f"- Yeast to Flour Ratio: {ratios['yeast_ratio']:.2f} (Ideal: ~0.20)")
                report.append(f"- Salt to Flour Ratio: {ratios['salt_ratio']:.2f} (Ideal: ~0.10)")
                
                if ratios['sugar_ratio'] > 1.0:
                    report.append("⚠️ High sugar content - may result in overly sweet product")
                if ratios['yeast_ratio'] < 0.1:
                    report.append("⚠️ Low yeast content - product may not rise properly")
                if ratios['salt_ratio'] > 0.3:
                    report.append("🚨 Excessive salt - will negatively impact taste")
        
        elif product_type == 'confectionery':
            report.append("**Analysis for Confectionery:**")
            if sugar == 0:
                report.append("⚠️ **CRITICAL:** No sugar detected - cannot produce confectionery!")
            else:
                report.append(f"- Sugar: {sugar:.1f}kg (Primary ingredient)")
                if flour > sugar * 0.1:
                    report.append("⚠️ Flour content too high for confectionery products")
                if yeast > 0:
                    report.append("⚠️ Yeast not typically needed for confectionery")
                if salt > sugar * 0.1:
                    report.append("⚠️ Salt content may be too high for sweet products")
        
        elif product_type == 'beverages':
            report.append("**Analysis for Beverages:**")
            if flour > 0:
                report.append("⚠️ Flour should not be used in beverage production")
            if sugar > 0:
                report.append(f"- Sugar content: {sugar:.1f}kg")
                if sugar > 100:  # Arbitrary threshold for beverages
                    report.append("⚠️ Very high sugar content for beverages")
            if yeast > 0:
                report.append("ℹ️ Yeast detected - possibly fermented beverage")
        
        elif product_type == 'dairy':
            report.append("**Analysis for Dairy Products:**")
            if flour > sugar:
                report.append("⚠️ Flour content unusually high for dairy products")
            if yeast > 0:
                report.append("✅ Yeast/culture content appropriate for fermented dairy")
        
        report.append("")
        
        # Process parameter analysis
        report.append(f"### Process Parameter Analysis")
        
        # Temperature analysis
        if product_info['requires_oven'] and oven_temp > 0:
            if oven_temp > 300:
                report.append(f"🚨 **DANGER:** Oven temperature ({oven_temp}°C) is extremely high!")
                report.append("   Risk of burning, fire hazard, and complete product failure.")
            elif oven_temp > 250:
                report.append(f"⚠️ Oven temperature ({oven_temp}°C) is very high - risk of burning")
            elif oven_temp < 150:
                report.append(f"⚠️ Oven temperature ({oven_temp}°C) may be too low for proper baking")
            else:
                report.append(f"✅ Oven temperature ({oven_temp}°C) is within acceptable range")
        elif not product_info['requires_oven'] and oven_temp > 100:
            report.append(f"⚠️ Oven heating ({oven_temp}°C) not necessary for this product type")
        
        # Equipment analysis
        mixer_speed = input_data['Mixer Speed (RPM)'].iloc[0]
        if total_mass > 100 and mixer_speed > 200:
            report.append(f"⚠️ Mixer speed ({mixer_speed} RPM) may be too high for large batch")
        elif mixer_speed < 50:
            report.append(f"⚠️ Mixer speed ({mixer_speed} RPM) may be too low for proper mixing")
        
        report.append("")
        
        # Recommendations
        report.append(f"### Recommendations")
        recommendations = []
        
        if quality_score < 50:
            recommendations.append("🚨 **IMMEDIATE ACTION REQUIRED:** Quality is below acceptable standards")
            recommendations.append("   Consider stopping production and reviewing all parameters")
        
        if product_type == 'baked_goods':
            if ratios['salt_ratio'] > 0.5:
                recommendations.append("• Reduce salt content significantly")
            if ratios['yeast_ratio'] < 0.05:
                recommendations.append("• Increase yeast content for proper fermentation")
            if oven_temp > 250:
                recommendations.append("• Reduce oven temperature to prevent burning")
        
        elif product_type == 'confectionery':
            if flour > 0:
                recommendations.append("• Remove or significantly reduce flour content")
            if yeast > 0:
                recommendations.append("• Remove yeast - not needed for most confectionery")
        
        elif product_type == 'beverages':
            if flour > 0:
                recommendations.append("• Remove flour completely - unsuitable for beverages")
            if oven_temp > 50:
                recommendations.append("• Reduce or eliminate heating - beverages don't require baking")
        
        # Scale-specific recommendations
        if total_mass > 1000:
            recommendations.append("• For industrial scale: Ensure proper mixing time and equipment capacity")
            recommendations.append("• Monitor temperature distribution across large batches")
            recommendations.append("• Consider batch subdivision for better quality control")
        
        if not recommendations:
            recommendations.append("✅ All parameters appear to be within acceptable ranges")
            recommendations.append("✅ Continue with current process settings")
        
        for rec in recommendations:
            report.append(rec)
        
        report.append("")
        report.append("---")
        report.append("*This report is generated by AI-powered F&B Quality Prediction System*")
        
        return "\n".join(report)
    
    def detect_anomalies(self, features):
        """Detect anomalies using Isolation Forest"""
        anomaly_scores = self.anomaly_detector.decision_function(features)
        anomaly_labels = self.anomaly_detector.predict(features)
        return anomaly_scores, anomaly_labels
        """Detect anomalies using Isolation Forest"""
        anomaly_scores = self.anomaly_detector.decision_function(features)
        anomaly_labels = self.anomaly_detector.predict(features)
        return anomaly_scores, anomaly_labels
    
    def prepare_features(self, df):
        """Prepare features for training"""
        # Calculate additional features
        df_features = df.copy()
        
        # Calculate ratios
        for idx, row in df_features.iterrows():
            ratios = self.calculate_ingredient_ratios(
                row['Flour (kg)'], row['Sugar (kg)'], 
                row['Yeast (kg)'], row['Salt (kg)']
            )
            df_features.loc[idx, 'Sugar_Ratio'] = ratios['sugar_ratio']
            df_features.loc[idx, 'Yeast_Ratio'] = ratios['yeast_ratio']
            df_features.loc[idx, 'Salt_Ratio'] = ratios['salt_ratio']
        
        # Calculate deviations from ideal
        for param, ideal_val in self.base_ideal_conditions.items():
            if param in df_features.columns:
                df_features[f'{param}_Deviation'] = abs(df_features[param] - ideal_val) / ideal_val
        
        # Select feature columns
        feature_columns = [
            'Flour (kg)', 'Sugar (kg)', 'Yeast (kg)', 'Salt (kg)',
            'Mixer Speed (RPM)', 'Water Temp (C)', 'Mixing Temp (C)',
            'Fermentation Temp (C)', 'Oven Temp (C)', 'Oven Humidity (%)',
            'Sugar_Ratio', 'Yeast_Ratio', 'Salt_Ratio'
        ]
        
        # Add deviation features
        deviation_features = [col for col in df_features.columns if '_Deviation' in col]
        feature_columns.extend(deviation_features)
        
        return df_features[feature_columns]
    
    def train_model(self, df):
        """Train the quality prediction model with realistic constraints to prevent overfitting"""
        # Calculate quality scores for training data
        quality_data = df.apply(self.calculate_realistic_quality_score, axis=1, result_type='expand')
        if isinstance(quality_data, pd.Series):
            df['Quality_Score'] = quality_data
        else:
            df['Quality_Score'] = quality_data[0]  # Take quality score, ignore product type
        
        # Add realistic noise to make the model more robust
        # In real manufacturing, there's always some variability
        np.random.seed(42)
        noise_factor = 0.15  # Increased to 15% noise to simulate real-world variability
        
        # Add multiple sources of noise to simulate real manufacturing
        base_noise = np.random.normal(0, 3, len(df))  # Base measurement noise
        temp_noise = np.random.normal(0, 2, len(df))  # Temperature variation noise
        process_noise = np.random.normal(0, 1.5, len(df))  # Process variation noise
        
        total_noise = base_noise + temp_noise + process_noise
        df['Quality_Score_Noisy'] = np.clip(df['Quality_Score'] + total_noise, 0, 100)
        
        # Add some random quality variations based on batch effects
        batch_effect = np.random.normal(0, 2, len(df))
        df['Quality_Score_Noisy'] = np.clip(df['Quality_Score_Noisy'] + batch_effect, 0, 100)
        
        # Prepare features
        X = self.prepare_features(df)
        y = df['Quality_Score_Noisy']  # Use noisy version for training
        
        # Split data with stratification to ensure diverse samples
        try:
            # Try stratified split first
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, 
                stratify=pd.cut(y, bins=5, labels=False, duplicates='drop')
            )
        except ValueError:
            # Fall back to regular split if stratification fails
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model with regularization to prevent overfitting
        self.model = RandomForestRegressor(
            n_estimators=30,          # Further reduced
            max_depth=4,              # Further reduced from 6
            min_samples_split=15,     # Increased
            min_samples_leaf=8,       # Increased
            max_features='sqrt',      # Use sqrt of features to add randomness
            bootstrap=True,           # Enable bootstrap sampling
            oob_score=True,          # Out-of-bag scoring for validation
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Perform cross-validation for more robust evaluation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, 
                                   cv=KFold(n_splits=5, shuffle=True, random_state=42),
                                   scoring='r2')
        
        # Train anomaly detector
        self.anomaly_detector.fit(X_train_scaled)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        y_train_pred = self.model.predict(X_train_scaled)
        
        # Calculate comprehensive metrics including overfitting detection
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_pred)
        overfitting_gap = train_r2 - test_r2
        
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': test_r2,
            'train_mse': mean_squared_error(y_train, y_train_pred),
            'train_r2': train_r2,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist(),
            'oob_score': self.model.oob_score_ if hasattr(self.model, 'oob_score_') else None,
            'overfitting_gap': overfitting_gap,
            'is_overfitted': overfitting_gap > 0.15,  # More lenient threshold
            'generalization_score': cv_scores.mean(),  # Use CV mean as generalization metric
            'feature_importance': dict(zip(X.columns, self.model.feature_importances_))
        }
        
        return metrics, X_test, y_test, y_pred
    
    def predict_quality(self, input_data):
        """Predict quality for new data"""
        if self.model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        # Prepare features
        X = self.prepare_features(input_data)
        X_scaled = self.scaler.transform(X)
        
        # Predict quality
        quality_pred = self.model.predict(X_scaled)
        
        # Detect anomalies
        anomaly_scores, anomaly_labels = self.detect_anomalies(X_scaled)
        
        # Calculate realistic quality scores
        realistic_scores = input_data.apply(self.calculate_realistic_quality_score, axis=1)
        
        return {
            'predicted_quality': quality_pred,
            'realistic_quality': realistic_scores,
            'anomaly_scores': anomaly_scores,
            'is_anomaly': anomaly_labels == -1,
            'feature_values': X
        }
    
    def save_model(self, filepath):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'anomaly_detector': self.anomaly_detector,
            'base_ideal_conditions': self.base_ideal_conditions,
            'acceptable_ranges': self.acceptable_ranges,
            'product_types': self.product_types
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.anomaly_detector = model_data['anomaly_detector']
        self.base_ideal_conditions = model_data.get('base_ideal_conditions', self.base_ideal_conditions)
        self.acceptable_ranges = model_data['acceptable_ranges']
        self.product_types = model_data.get('product_types', self.product_types)


def load_and_preprocess_data(filepath):
    """Load and preprocess the F&B data"""
    df = pd.read_excel(filepath)
    
    # Basic data cleaning
    df = df.dropna()
    
    # Remove outliers using IQR method for each parameter
    for column in df.select_dtypes(include=[np.number]).columns:
        if column not in ['Batch_ID', 'Time']:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return df


if __name__ == "__main__":
    # Load data - check multiple possible paths
    data_paths = ["original_data.xlsx", "../original_data.xlsx"]
    df = None
    
    for path in data_paths:
        try:
            df = load_and_preprocess_data(path)
            print(f"Data loaded from: {path}")
            break
        except FileNotFoundError:
            continue
    
    if df is None:
        print("Error: Could not find original_data.xlsx")
        exit(1)
    
    # Initialize and train model
    predictor = QualityPredictor()
    metrics, X_test, y_test, y_pred = predictor.train_model(df)
    
    print("Model Training Complete!")
    print(f"R² Score: {metrics['r2']:.4f}")
    print(f"Mean Absolute Error: {metrics['mae']:.4f}")
    print(f"Mean Squared Error: {metrics['mse']:.4f}")
    
    # Save model
    predictor.save_model("fnb_quality_model.pkl")
    print("Model saved successfully!")
