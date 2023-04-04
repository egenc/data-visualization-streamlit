"""This script creates a local WebApp for EDA and interactive data manipulation"""
import base64
import datetime
import io
import pandas as pd
import streamlit as st
from plotly import graph_objs as go



DATA_PATH = "data/weatherlink_data_on_site_updated.parquet"

st.set_page_config(page_title="Data Visualization",
                   page_icon=None,
                   layout="centered",
                   initial_sidebar_state="auto",
                   menu_items=None)

st.write("Hello :wave:")
st.write("In this simple WebApp, I will try to do how to EDA and manipulate data interactively :dart:")
st.write("Here is my [Github account](https://github.com/egenc) to give you a glimpse what kind of projects I am working on.")

with st.sidebar:
    st.markdown("[#1. Checking counts of :blue[NaN values]:](Section 1)", unsafe_allow_html=True)


@st.cache_data
def load_data(path):
    """loads input data"""
    return pd.read_parquet(path)

df = load_data(DATA_PATH)

st.title(':seedling: Data Visualization with Streamlit :seedling:')

st.write("Data looks like this: (click top right icon to expand)")
st.write(df.head())

st.write("---")
st.subheader(':exclamation: _Some Insights About Data_ :exclamation:')

col1, col2 = st.columns(2)

with col1:
    op_type = st.radio(
        "What would you like to know about data :point_down:",
        options=["columns", "description", "info", "shape", "none"],
    )

with col2:
    df_ops = {"columns": df.columns, 
              "description": df.describe(),
              "shape": f"(num_rows, num_columns): **{df.shape}**" ,
              "none": "Please select a box on the left to get insights about data"}

    if op_type == "info":

        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
    else:
        st.markdown("**Slide left-right to explore:**")
        st.write(df_ops[op_type])
st.write("---")
st.subheader("1. Checking counts of :blue[NaN values]:")
st.write(df.isnull().sum())

st.write("---")
st.write("**For a start, let's drop columns with too many :blue[NaN values]:**")
st.write("Do you agree to drop columns **['2nd Temp - °C', 'High Wind Direction', 'Wind Direction']** with too many NaN values:")
agree = st.checkbox('I agree')

if agree:
    df.drop(['2nd Temp - °C', 'High Wind Direction', 'Wind Direction'], axis=1, inplace=True)

st.write(df.head())

st.write("---")
st.write("Let's check counts of :blue[NaN values] again:")
st.write(df.isnull().sum())

col1, col2 = st.columns(2)

with col1:
    nan_op = st.radio(
            "How would you like to handle NaN values :point_down:",
            options=["drop rows with NaN", "Replace NaNs with a numerical value"],
        )

with col2:
    if nan_op == "drop rows with NaN":
        df.dropna(inplace=True)
    else:
        number = st.number_input('Insert a number to replace NaNs:', min_value=0)
        df.fillna(number, inplace=True)

    st.write("**Let's check counts of :blue[NaN values] one last time:**")
    st.dataframe(df.isnull().sum())

st.write("Average Windspeed: :tornado:", df["Wind Speed - km/h"].mean())
st.write("Average Temperature: :mostly_sunny:", df["Temp - °C"].mean())

st.write("---")
st.subheader("2. Checking :blue[Repeating Values]")

col1, col2 = st.columns(2)
with col1:
    N_repetative = int(st.number_input('Amount of repetative values: (i. e. 5)',
                       max_value=len(df), min_value=4))
    col = st.radio(
        "Please select a column to check repetative values :point_down:",
        options=df.columns,
    )

@st.cache_data
def get_value_counts(d_frame):
    """gets value counts of a specific column"""
    counts = d_frame.value_counts()
    return counts[counts > N_repetative].index.tolist()

with col2:
    matches = get_value_counts(df[col])
    st.write(f"**__Values in :red[{col}] that occur more than :red[{N_repetative}] times:__** {matches}")

st.write("---")
st.subheader(":hourglass_flowing_sand: 3. Checking Timestamps :hourglass_flowing_sand:")

st.write("- Duplicate TimeStamps (if exists):")
duplicates_mask = df.duplicated(['DateTime'], keep=False)

st.write(df[duplicates_mask])

st.write("- Select Dates: :warning:(Please pick time carefully as dataframe will change accordingly and so are results):warning:")

col1, col2 = st.columns(2)

with col1:
    d_min = st.date_input(
        ":red[Start Date]",
        datetime.date(2020, 11, 3),
        key="start_date")
    t_min = st.time_input(':red[Start time]', datetime.time(16, 00),
        key="start_time")
    dt_min = datetime.datetime.combine(d_min, t_min)

with col2:
    d_max = st.date_input(
        ":blue[End Date]",
        datetime.datetime.now(),
        key="end_date")

    t_max = st.time_input(':blue[End time]',
                          datetime.time(00, 00),
                          key="end_time")

    dt_max = datetime.datetime.combine(d_max, t_max)

mask = (df['DateTime'] >= dt_min) & (df['DateTime'] <= dt_max)

df = df.loc[mask]

st.write("---")
ticked = st.checkbox('I want to see statistics of data between selected dates.')

if ticked:
    col1, col2 = st.columns(2)

    with col1:
        op_type = st.radio(
            "Explore data👉",
            options=["columns", "description", "info", "shape", "none"],
        )

    with col2:
        df_ops = {"columns": df.columns, 
                "description": df.describe(),
                "shape": f"(num_rows, num_columns): **{df.shape}**" ,
                "none": "Please select a box on the left to get insights about data"}

        if op_type == "info":

            buffer = io.StringIO()
            df.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)
        else:
            st.markdown("**Slide left-right to explore:**")
            st.write(df_ops[op_type])

st.write("---")
st.write("- Check outliers:")

col1, col2 = st.columns(2)
with col1:
    N_repetative = int(st.number_input('Amount of std from mean (i. e. 5)',
                       max_value=30,
                       min_value=1))
    col = st.radio(
        "Please select a column to check std-mean relation👉",
        options=df.columns,
    )

with col2:
    try:
        column = df[col]
        mean = column.mean()
        std = column.std()

        outliers = column[(column > mean + N_repetative * std) | (column < mean - N_repetative * std)]

        st.write("Outlier values:")
        st.write(outliers)
    except TypeError:
        pass


# Add a download button to download the dataframe as a CSV file
csv = df.to_csv(index=False)
b64 = base64.b64encode(csv.encode()).decode()
st.download_button(
    label=":inbox_tray: Download latest CSV with changes :inbox_tray:",
    data=csv,
    file_name='saved_csv.csv',
    mime='text/csv'
)

st.write("---")
# Plot raw data
def plot_raw_data(cols_list):
    """Plots columns based on Time"""
    fig = go.Figure()
    for s_col in cols_list:
        fig.add_trace(go.Scatter(x=df['DateTime'], y=df[s_col], name=f"Time over {s_col}"))
    fig.layout.update(title_text=f'Time Series data with column {cols_list}',
                      xaxis_rangeslider_visible=True)
    return fig

st.write("---")

options = st.multiselect(
                        'please select the columns to plot',
                        df.columns,
                        ['Temp - °C', 'Wind Speed - km/h']
                        )

st.write("Please expand the small chart below to zoom in & out")
fig = plot_raw_data(options)
st.plotly_chart(fig)

# Create an in-memory buffer
buffer = io.BytesIO()

# Save the figure as a pdf to the buffer
fig.write_image(file=buffer, format="pdf")

# Download the pdf from the buffer
st.download_button(
    label=":inbox_tray: Download graph as PDF :inbox_tray:",
    data=buffer,
    file_name="results/figure.pdf",
    mime="application/pdf",
)

st.write("---")
# Compute the correlation matrix
corr_matrix = df.corr(numeric_only=True)

# Create a heatmap plot of the correlation matrix using plotly
fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        reversescale=True,
        zmin=-1,
        zmax=1))

# Customize the plot layout
fig.update_layout(
    title="Correlation Matrix",
    xaxis_title="Features",
    yaxis_title="Features",
    width=1000,  # Set the width of the figure
    height=1000,  # Set the height of the figure
    margin=dict(l=40, r=40, b=40, t=40),
    paper_bgcolor="black",
)

# Display the plot
st.plotly_chart(fig)
