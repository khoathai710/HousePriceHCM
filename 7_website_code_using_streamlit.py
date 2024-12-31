import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import HistGradientBoostingRegressor

# Load the data
data = pd.read_csv('https://raw.githubusercontent.com/KhiemDangLe/Final-Project/main/DataFolder/5_preprocessed_data.csv')
df_location = pd.read_csv('https://raw.githubusercontent.com/KhiemDangLe/Final-Project/main/DataFolder/7_coordinates_by_street_name_1_5000.csv')
df_location2 = pd.read_csv('https://raw.githubusercontent.com/KhiemDangLe/Final-Project/main/DataFolder/8_coordinates_by_street_nam_5000_end.csv')

# Merge datasets
merged_df1 = pd.merge(data, df_location[['article_id', 'longitude', 'latitude']], on='article_id', how='inner')
merged_df2 = pd.merge(data, df_location2[['article_id', 'longitude', 'latitude']], on='article_id', how='inner')
merged_df = pd.concat([merged_df1, merged_df2], ignore_index=True)

# Split data into features and target variable
y = merged_df['price'].copy()
X = merged_df.drop('price', axis=1).copy()

# Selected columns
selected_int_cols = ['area', 'bedroom', 'wc', 'numbers_of_floors', 'count_conveniences']
selected_cat_cols = ['district', 'direction', 'has_rooftop', 'total_room_LLm', 'furnished']

# Function to convert columns to categorical
def to_categorical(df):
    return pd.DataFrame(df).astype('category')

# ColumnTransformer for different columns
transform_type_cols = ColumnTransformer([
    ('to_categorical', FunctionTransformer(to_categorical), selected_cat_cols),
    ('int_cols', 'passthrough', selected_int_cols)
], remainder='drop')

# Create pipeline
hist_native_pipeline = Pipeline([
    ('transform_type_cols', transform_type_cols),
    ('model', HistGradientBoostingRegressor(loss='squared_error', random_state=42, early_stopping=False))
])

# Set best parameters
best_params = {
    'model__learning_rate': 0.075,
    'model__max_iter': 80,
    'model__max_depth': 13,
    'model__min_samples_leaf': 116,
    'model__categorical_features': range(len(selected_cat_cols))
}
hist_native_pipeline.set_params(**best_params)

# Train pipeline
hist_native_pipeline.fit(X, y)

# Streamlit app
st.set_page_config(page_title="House Price Prediction", page_icon="🏠")
st.title("🏠 House Price Prediction")

# Custom background and CSS
page_bg_img = '''
<style>
body {
    background-image: url("https://your-static-url.com/023-vietnamese-town-house-mm-architects-1390x927.jpg");
    background-size: cover;
    color: #333;
    font-family: 'Roboto', sans-serif;
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
    margin: 0;
}
.stApp {
    background: rgba(255, 255, 255, 0.9);
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
    text-align: center;
}
h1 {
    font-size: 2.5em;
    margin-bottom: 20px;
    color: #2c7873;
}
label {
    font-weight: bold;
}
.stButton>button {
    background-color: #2c7873;
    color: #fff;
    border: none;
    padding: 10px 20px;
    font-size: 1em;
    border-radius: 5px;
    transition: background-color 0.3s ease, transform 0.3s ease;
}
.stButton>button:hover {
    background-color: #1e5550;
    transform: scale(1.05);
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Sidebar for input
st.sidebar.header("Input Information")
st.sidebar.text("Please enter your details and click 'Predict' to see the result.")

# Input fields
input_data = {}
input_data['area'] = st.sidebar.text_input('Area (m²)', value="")
input_data['district'] = st.sidebar.selectbox('District', X['district'].unique())
input_data['bedroom'] = st.sidebar.number_input('Number of Bedrooms', min_value=0, step=1, format="%d")
input_data['wc'] = st.sidebar.number_input('Number of Bathrooms', min_value=0, step=1, format="%d")
input_data['numbers_of_floors'] = st.sidebar.number_input('Number of Floors', min_value=0, step=1, format="%d")
input_data['count_conveniences'] = 0  # Set default value and hide the field
input_data['direction'] = st.sidebar.selectbox('Direction', X['direction'].unique())
input_data['has_rooftop'] = st.sidebar.checkbox('Rooftop', value=False)
input_data['total_room_LLm'] = st.sidebar.number_input('Total Rooms', min_value=0, step=1, format="%d")
input_data['furnished'] = st.sidebar.checkbox('Furnished', value=False)

# Predict button
if st.sidebar.button('Predict'):
    try:
        # Check if all required fields are filled
        if not input_data['area'] or input_data['area'].strip() == '' or float(input_data['area']) <= 0:
            raise ValueError("Please fill in the area correctly.")
        elif input_data['wc'] <= 0:
            raise ValueError("Please fill in the number of bathrooms correctly.")
        
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Make prediction
        predicted_price = hist_native_pipeline.predict(input_df)

        # Display result with custom CSS
        result_css = '''
        <style>
        body {
            background: url('static/OIP.jpg') no-repeat center center fixed;
            background-size: cover;
            font-family: 'Roboto', sans-serif;
            color: #333;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }
        .container {
            background: rgba(255, 255, 255, 0.9);
            padding: 30px 50px;
            border-radius: 15px;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
            text-align: center;
        }
        h1 {
            font-size: 2.5em;
            margin-bottom: 20px;
            color: #2c7873;
        }
        h2 {
            font-size: 2em;
            margin-bottom: 20px;
            color: #2c7873;
        }
        h1 .fas {
            margin-right: 10px;
        }
        a {
            color: #2c7873;
            text-decoration: none;
            font-size: 1.2em;
        }
        a:hover {
            text-decoration: underline;
        }
        </style>
        '''
        st.markdown(result_css, unsafe_allow_html=True)
        st.markdown(f'''
        <div class="container">
            <h1><i class="fas fa-chart-line"></i> Prediction Result</h1>
            <h2>Predicted House Price: {predicted_price[0]:,.2f} billion VND</h2>
            <br><br>
            <a href="/" class="btn btn-link">Go back to home page</a>
        </div>
        ''', unsafe_allow_html=True)
    
    except Exception as e:
        # Display error with custom CSS
        error_css = '''
        <style>
        body {
            background: #ffcccc;
            font-family: 'Roboto', sans-serif;
            color: #333;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }
        .container {
            background: rgba(255, 255, 255, 0.9);
            padding: 30px 50px;
            border-radius: 15px;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
            text-align: center;
        }
        h1 {
            font-size: 2.5em;
            margin-bottom: 20px;
            color: #d9534f;
        }
        p {
            font-size: 1.5em;
            margin-bottom: 20px;
        }
        a {
            color: #d9534f;
            text-decoration: none;
            font-size: 1.2em;
        }
        a:hover {
            text-decoration: underline;
        }
        </style>
        '''
        st.markdown(error_css, unsafe_allow_html=True)
        st.markdown(f'''
        <div class="container">
            <h1>Error</h1>
            <p>{str(e)}</p>
            <br><br>
            <a href="/">Go back to home page</a>
        </div>
        ''', unsafe_allow_html=True)