import pandas as pd
import numpy as np
from quality_predictor import QualityPredictor

def create_test_scenarios():
    """Create various test scenarios to demonstrate the system"""
    
    scenarios = [
        {
            'name': 'Ideal Conditions',
            'description': 'Perfect baking conditions',
            'Flour (kg)': 10.0,
            'Sugar (kg)': 5.0,
            'Yeast (kg)': 2.0,
            'Salt (kg)': 1.0,
            'Mixer Speed (RPM)': 150,
            'Water Temp (C)': 25,
            'Mixing Temp (C)': 35,
            'Fermentation Temp (C)': 35,
            'Oven Temp (C)': 180,
            'Oven Humidity (%)': 45
        },
        {
            'name': 'Too Much Salt',
            'description': 'Excessive salt content (disaster scenario)',
            'Flour (kg)': 10.0,
            'Sugar (kg)': 5.0,
            'Yeast (kg)': 2.0,
            'Salt (kg)': 8.0,  # 8x normal amount
            'Mixer Speed (RPM)': 150,
            'Water Temp (C)': 25,
            'Mixing Temp (C)': 35,
            'Fermentation Temp (C)': 35,
            'Oven Temp (C)': 180,
            'Oven Humidity (%)': 45
        },
        {
            'name': 'Extreme Oven Temperature',
            'description': 'Very high oven temperature',
            'Flour (kg)': 10.0,
            'Sugar (kg)': 5.0,
            'Yeast (kg)': 2.0,
            'Salt (kg)': 1.0,
            'Mixer Speed (RPM)': 150,
            'Water Temp (C)': 25,
            'Mixing Temp (C)': 35,
            'Fermentation Temp (C)': 35,
            'Oven Temp (C)': 300,  # Very high temperature
            'Oven Humidity (%)': 45
        },
        {
            'name': 'No Yeast',
            'description': 'Missing yeast (bread won\'t rise)',
            'Flour (kg)': 10.0,
            'Sugar (kg)': 5.0,
            'Yeast (kg)': 0.0,  # No yeast
            'Salt (kg)': 1.0,
            'Mixer Speed (RPM)': 150,
            'Water Temp (C)': 25,
            'Mixing Temp (C)': 35,
            'Fermentation Temp (C)': 35,
            'Oven Temp (C)': 180,
            'Oven Humidity (%)': 45
        },
        {
            'name': 'Balanced Recipe',
            'description': 'Good proportions with slight variations',
            'Flour (kg)': 12.0,
            'Sugar (kg)': 6.0,
            'Yeast (kg)': 2.4,
            'Salt (kg)': 1.2,
            'Mixer Speed (RPM)': 160,
            'Water Temp (C)': 27,
            'Mixing Temp (C)': 37,
            'Fermentation Temp (C)': 36,
            'Oven Temp (C)': 185,
            'Oven Humidity (%)': 47
        },
        {
            'name': 'Low Quality Scenario',
            'description': 'Multiple parameter deviations',
            'Flour (kg)': 8.0,   # Low flour
            'Sugar (kg)': 12.0,  # Too much sugar
            'Yeast (kg)': 0.5,   # Too little yeast
            'Salt (kg)': 3.0,    # Too much salt
            'Mixer Speed (RPM)': 80,  # Low mixing speed
            'Water Temp (C)': 50,     # High water temp
            'Mixing Temp (C)': 50,    # High mixing temp
            'Fermentation Temp (C)': 50,  # High fermentation temp
            'Oven Temp (C)': 220,     # High oven temp
            'Oven Humidity (%)': 70   # High humidity
        }
    ]
    
    return scenarios

def test_quality_predictor():
    """Test the quality predictor with various scenarios"""
    predictor = QualityPredictor()
    scenarios = create_test_scenarios()
    
    print("=== F&B Quality Prediction Test Results ===\n")
    
    for scenario in scenarios:
        # Create input dataframe
        input_data = pd.DataFrame({
            'Flour (kg)': [scenario['Flour (kg)']],
            'Sugar (kg)': [scenario['Sugar (kg)']],
            'Yeast (kg)': [scenario['Yeast (kg)']],
            'Salt (kg)': [scenario['Salt (kg)']],
            'Mixer Speed (RPM)': [scenario['Mixer Speed (RPM)']],
            'Water Temp (C)': [scenario['Water Temp (C)']],
            'Mixing Temp (C)': [scenario['Mixing Temp (C)']],
            'Fermentation Temp (C)': [scenario['Fermentation Temp (C)']],
            'Oven Temp (C)': [scenario['Oven Temp (C)']],
            'Oven Humidity (%)': [scenario['Oven Humidity (%)']]
        })
        
        # Calculate quality score
        quality_score = predictor.calculate_realistic_quality_score(input_data.iloc[0])
        
        # Calculate ratios
        ratios = predictor.calculate_ingredient_ratios(
            scenario['Flour (kg)'], scenario['Sugar (kg)'],
            scenario['Yeast (kg)'], scenario['Salt (kg)']
        )
        
        print(f"Scenario: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print(f"Quality Score: {quality_score:.1f}%")
        print(f"Sugar Ratio: {ratios['sugar_ratio']:.2f} (ideal: 0.50)")
        print(f"Yeast Ratio: {ratios['yeast_ratio']:.2f} (ideal: 0.20)")
        print(f"Salt Ratio: {ratios['salt_ratio']:.2f} (ideal: 0.10)")
        
        # Quality assessment
        if quality_score >= 85:
            status = "Excellent"
        elif quality_score >= 70:
            status = "Good"
        elif quality_score >= 50:
            status = "Warning"
        else:
            status = "Poor"
        
        print(f"Status: {status}")
        print("-" * 50)

def generate_edge_case_data():
    """Generate edge case scenarios for testing"""
    edge_cases = [
        # Extreme salt scenario (as mentioned in requirements)
        {
            'name': 'Extreme Salt (100kg salt with 9kg flour)',
            'Flour (kg)': 9.0,
            'Sugar (kg)': 5.0,
            'Yeast (kg)': 2.0,
            'Salt (kg)': 100.0,  # Catastrophic amount
        },
        # Extreme temperature scenario (1000 degree oven)
        {
            'name': 'Extreme Temperature (1000°C oven)',
            'Flour (kg)': 10.0,
            'Sugar (kg)': 5.0,
            'Yeast (kg)': 2.0,
            'Salt (kg)': 1.0,
            'Oven Temp (C)': 1000.0,  # Catastrophic temperature
        },
        # Balanced high quantity scenario (2000kg flour, 1000kg sugar)
        {
            'name': 'Large Scale Good Ratios (2000kg flour, 1000kg sugar)',
            'Flour (kg)': 2000.0,
            'Sugar (kg)': 1000.0,  # 0.5 ratio - good!
            'Yeast (kg)': 400.0,   # 0.2 ratio - good!
            'Salt (kg)': 200.0,    # 0.1 ratio - good!
        }
    ]
    
    predictor = QualityPredictor()
    
    print("\n=== Edge Case Testing ===\n")
    
    for case in edge_cases:
        # Fill missing values with defaults
        case.setdefault('Sugar (kg)', 5.0)
        case.setdefault('Yeast (kg)', 2.0)
        case.setdefault('Salt (kg)', 1.0)
        case.setdefault('Mixer Speed (RPM)', 150)
        case.setdefault('Water Temp (C)', 25)
        case.setdefault('Mixing Temp (C)', 35)
        case.setdefault('Fermentation Temp (C)', 35)
        case.setdefault('Oven Temp (C)', 180)
        case.setdefault('Oven Humidity (%)', 45)
        
        # Create input dataframe
        input_data = pd.DataFrame([case])
        
        # Calculate quality score
        quality_score = predictor.calculate_realistic_quality_score(input_data.iloc[0])
        
        print(f"Edge Case: {case['name']}")
        print(f"Quality Score: {quality_score:.1f}%")
        print(f"Ingredients: Flour={case['Flour (kg)']}kg, Sugar={case['Sugar (kg)']}kg, "
              f"Yeast={case['Yeast (kg)']}kg, Salt={case['Salt (kg)']}kg")
        
        if case['name'].startswith('Large Scale'):
            ratios = predictor.calculate_ingredient_ratios(
                case['Flour (kg)'], case['Sugar (kg)'],
                case['Yeast (kg)'], case['Salt (kg)']
            )
            print(f"Ratios: Sugar={ratios['sugar_ratio']:.2f}, "
                  f"Yeast={ratios['yeast_ratio']:.2f}, "
                  f"Salt={ratios['salt_ratio']:.2f}")
            print("✅ Good ratios maintained even at large scale!")
        
        print("-" * 60)

if __name__ == "__main__":
    test_quality_predictor()
    generate_edge_case_data()
