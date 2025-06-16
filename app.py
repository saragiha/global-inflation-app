import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, NumeralTickFormatter
from bokeh.palettes import Category10
from bokeh.layouts import column
from bokeh.io import show

# --- PAGE CONFIG ---
st.set_page_config(page_title="Global Inflation Explorer", layout="wide")

# --- TITLE & DESCRIPTION ---
st.title("🌍 Global Inflation Explorer")
st.markdown("""
Explore historical and forecasted inflation rates for countries around the world. 
- **Select up to 5 countries** to compare their inflation trends.
- **Choose the chart type**: Line, Bar, or Multi-Line.
- **Hover** over the chart for detailed values.
""")

# --- LOAD DATA ---
@st.cache_data

def load_data():
    df = pd.read_csv("global_inflation_data.csv")
    df_long = df.melt(id_vars=['country_name'], var_name='year', value_name='inflation_rate')
    df_long = df_long[df_long['year'].astype(str).str.isdigit()]
    df_long['year'] = df_long['year'].astype(int)
    df_long = df_long.dropna()
    return df, df_long

df, df_long = load_data()
all_countries = df['country_name'].dropna().unique().tolist()

# --- SIDEBAR ---
st.sidebar.header("Options")
chart_type = st.sidebar.selectbox("Chart Type", [
    "Line Chart",
    "Bar Chart",
    "Multi-Line Chart",
    "Pie Chart",
    "Plot Graph",
    "Stacked Area Chart",
    "Horizontal Bar Chart",
    "Box Plot",
    "Average Trend Line"
])


# --- FORECASTING (for line chart) ---
@st.cache_data

def get_forecast_data(df_long, all_countries):
    forecast_years = [2025, 2026, 2027, 2028, 2029, 2030]
    window_size = 7
    data_dict = {}
    for country in all_countries:
        country_data = df_long[df_long['country_name'] == country].sort_values('year')
        years = country_data['year'].tolist()
        series = country_data['inflation_rate'].values
        if len(series) < window_size + 1:
            continue
        X, y, year_features = [], [], []
        for i in range(len(series) - window_size):
            X.append(series[i:i+window_size])
            y.append(series[i+window_size])
            year_features.append(years[i + window_size])
        X, y = np.array(X), np.array(y)
        year_features = np.array(year_features).reshape(-1, 1)
        X = np.hstack([X, year_features])
        model = XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=4,
                             subsample=0.8, colsample_bytree=0.8, gamma=1, random_state=42)
        model.fit(X, y)
        last_window = list(series[-window_size:])
        future_preds = []
        for year in forecast_years:
            input_window = np.array(last_window + [year]).reshape(1, -1)
            pred = max(model.predict(input_window)[0], 0)
            future_preds.append(pred)
            last_window = last_window[1:] + [pred]
        all_years = years + forecast_years
        all_values = list(series) + future_preds
        labels = ['historical'] * len(series) + ['forecast'] * len(future_preds)
        max_idx = np.argmax(all_values)
        max_val = all_values[max_idx]
        max_year = all_years[max_idx]
        data_dict[country] = {
            'year': all_years,
            'inflation': all_values,
            'label': labels,
            'max_inflation': max_val,
            'max_year': max_year
        }
    return data_dict

data_dict = get_forecast_data(df_long, all_countries)

# Sidebar: Pilihan negara dan tahun
selected_countries = st.sidebar.multiselect(
    "Select up to 5 Countries", all_countries, default=all_countries[:3], max_selections=5
)

show_forecast = st.sidebar.checkbox("Include Forecast Years (2025–2030)", value=True)

# Atur range tahun berdasarkan forecast
min_year = df_long['year'].min()
max_actual_year = df_long['year'].max()
max_year = 2030 if show_forecast else max_actual_year

start_year, end_year = st.sidebar.select_slider(
    "Select Year Range",
    options=list(range(min_year, max_year + 1)),
    value=(min_year, max_actual_year)
)

# Forecast data
if show_forecast:
    forecast_data = get_forecast_data(df_long, all_countries)

    # Gabungkan ke df_long
    additional_rows = []
    for country in forecast_data:
        rows = pd.DataFrame({
            'country_name': country,
            'year': forecast_data[country]['year'],
            'inflation_rate': forecast_data[country]['inflation'],
            'label': forecast_data[country]['label']
        })
        additional_rows.append(rows)
    df_forecast = pd.concat(additional_rows, ignore_index=True)
    df_combined = pd.concat([df_long.assign(label='historical'), df_forecast[df_forecast['label'] == 'forecast']], ignore_index=True)
else:
    df_combined = df_long.assign(label='historical')

# Filter berdasarkan tahun dan negara
df_filtered = df_combined[
    (df_combined['year'] >= start_year) &
    (df_combined['year'] <= end_year) &
    (df_combined['country_name'].isin(selected_countries))
]
def get_filtered_data():
    return df_combined[
        (df_combined['year'] >= start_year) &
        (df_combined['year'] <= end_year) &
        (df_combined['country_name'].isin(selected_countries))
    ]


def make_line_chart(selected_countries):
    df_filtered = get_filtered_data()
    country = selected_countries[0] if selected_countries else all_countries[0]
    data = df_filtered[df_filtered['country_name'] == country]
    source = ColumnDataSource(data={
        'year': data['year'],
        'inflation': data['inflation_rate'],
        'label': data['label']
    })
    p = figure(title=f"Inflation Forecast for {country}", x_axis_label='Year', y_axis_label='Inflation Rate (%)',
               width=900, height=400, tools="pan,wheel_zoom,reset")
    p.line(x='year', y='inflation', source=source, line_width=2, color="gray", legend_label="Inflation Rate")
    p.scatter(x='year', y='inflation', source=source, size=6, fill_color='label', marker='circle', legend_field='label')
    p.add_tools(HoverTool(tooltips=[("Year", "@year"), ("Inflation (%)", "@inflation{0.00}"), ("Data Type", "@label")]))
    p.xaxis.major_label_orientation = "vertical"
    p.yaxis.formatter = NumeralTickFormatter(format="0.00")
    p.grid.grid_line_alpha = 0.3
    p.legend.location = "top_left"
    p.xaxis.ticker = data['year']
    return p

def make_bar_chart(selected_countries):
    df_filtered = get_filtered_data()
    countries = selected_countries if selected_countries else all_countries[:3]
    bar_data = df_filtered[df_filtered['country_name'].isin(countries)]
    agg_data = bar_data.groupby('country_name').agg(
        max_inflation=('inflation_rate', 'max'),
        max_year=('year', lambda x: bar_data.loc[x.idxmax(), 'year'])
    ).reset_index()
    source = ColumnDataSource(agg_data)
    bar = figure(x_range=agg_data['country_name'], title="Maximum Inflation Rate and Year by Country",
                 x_axis_label="Country", y_axis_label="Max Inflation Rate (%)", width=900, height=400,
                 tools="pan,wheel_zoom,reset")
    bar.vbar(x='country_name', top='max_inflation', width=0.6, source=source, color="darkorange")
    bar.xaxis.major_label_orientation = 1.0
    bar.yaxis.formatter = NumeralTickFormatter(format="0.00")
    bar.add_tools(HoverTool(tooltips=[
        ("Country", "@country_name"),
        ("Max Inflation", "@max_inflation{0.00}"),
        ("Max Year", "@max_year")
    ]))
    return bar


def make_multi_line_chart(selected_countries):
    df_filtered = get_filtered_data()
    countries = selected_countries if selected_countries else all_countries[:5]
    colors = Category10[10]
    multi_line_data = {'year': [], 'inflation': [], 'country': [], 'color': []}
    point_data = {'year': [], 'inflation': [], 'country': [], 'color': []}
    for idx, country in enumerate(countries):
        country_data = df_filtered[df_filtered['country_name'] == country].sort_values('year')
        years = country_data['year'].tolist()
        inflations = country_data['inflation_rate'].tolist()
        multi_line_data['year'].append(years)
        multi_line_data['inflation'].append(inflations)
        multi_line_data['country'].append(country)
        multi_line_data['color'].append(colors[idx % len(colors)])
        point_data['year'].extend(years)
        point_data['inflation'].extend(inflations)
        point_data['country'].extend([country] * len(years))
        point_data['color'].extend([colors[idx % len(colors)]] * len(years))
    multi_line_source = ColumnDataSource(multi_line_data)
    point_source = ColumnDataSource(point_data)
    p = figure(title="Top Selected Countries: Inflation Rate Over Time",
               x_axis_label='Year', y_axis_label='Inflation Rate (%)',
               width=900, height=500, tools="pan,wheel_zoom,reset")
    p.multi_line(xs='year', ys='inflation', legend_field='country',
                 line_color='color', line_width=2, source=multi_line_source)
    p.scatter(x='year', y='inflation', source=point_source, color='color', size=8, alpha=0)
    p.add_tools(HoverTool(tooltips=[
        ("Country", "@country"),
        ("Year", "@year"),
        ("Inflation Rate", "@inflation{0.00}")
    ], mode='mouse'))
    p.legend.location = "top_left"
    p.xaxis.axis_label_text_font_style = "italic"
    p.yaxis.axis_label_text_font_style = "italic"
    p.title.text_font_size = "16pt"
    return p


def make_pie_chart(selected_countries):
    from bokeh.transform import cumsum
    import math

    countries = selected_countries if selected_countries else all_countries[:3]
    pie_data = df_filtered[df_filtered['country_name'].isin(countries)]
    pie_data = pie_data.groupby('country_name')['inflation_rate'].mean().reset_index()
    pie_data['angle'] = pie_data['inflation_rate'] / pie_data['inflation_rate'].sum() * 2 * math.pi
    pie_data['color'] = Category10[len(pie_data)]
    source = ColumnDataSource(pie_data)
    p = figure(title="Average Inflation Rate (Pie Chart)", toolbar_location=None, tools="hover", tooltips="@country_name: @inflation_rate{0.00}", width=700, height=600)
    p.wedge(x=0, y=1, radius=0.4,
            start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
            line_color="white", fill_color='color', legend_field='country_name', source=source)
    p.axis.visible = False
    p.grid.visible = False
    return p

def make_dot_plot(selected_countries):
    data = df_filtered[df_filtered['country_name'].isin(selected_countries)]
    source = ColumnDataSource(data)
    p = figure(title="Inflation Dot Plot", x_axis_label="Year", y_axis_label="Inflation Rate (%)",
               width=900, height=400, tools="pan,wheel_zoom,reset")
    p.circle(x='year', y='inflation_rate', source=source, size=8, color='navy', alpha=0.6)
    p.add_tools(HoverTool(tooltips=[("Country", "@country_name"), ("Year", "@year"), ("Inflation", "@inflation_rate{0.00}")]))
    return p

def make_stacked_area_chart(selected_countries):
    pivot_df = df_filtered[df_filtered['country_name'].isin(selected_countries)].pivot(index='year', columns='country_name', values='inflation_rate').fillna(0)
    years = pivot_df.index.tolist()
    colors = Category10[10]
    p = figure(title="Stacked Area Chart", x_axis_label='Year', y_axis_label='Inflation Rate (%)', width=900, height=400, tools="pan,wheel_zoom,reset")
    last = np.zeros(len(pivot_df))
    for i, country in enumerate(pivot_df.columns):
        values = pivot_df[country].values
        p.varea(x=years, y1=last, y2=last + values, color=colors[i % 10], alpha=0.7, legend_label=country)
        last += values
    p.legend.location = "top_left"
    return p

def make_horizontal_bar_chart(selected_countries):
    data = df_filtered[df_filtered['country_name'].isin(selected_countries)]
    avg_inflation = data.groupby('country_name')['inflation_rate'].mean().reset_index()
    source = ColumnDataSource(avg_inflation)
    p = figure(y_range=avg_inflation['country_name'], width=700, height=400,
               title="Average Inflation Rate (Horizontal Bar)", toolbar_location=None, tools="hover", tooltips=[("Inflation", "@inflation_rate{0.00}")])
    p.hbar(y='country_name', right='inflation_rate', height=0.5, source=source, color="skyblue")
    p.xaxis.axis_label = "Average Inflation Rate (%)"
    return p

def make_box_plot(selected_countries):
    from bokeh.transform import jitter
    data = df_filtered[df_filtered['country_name'].isin(selected_countries)]
    p = figure(title="Inflation Rate Distribution (Box Plot Approx.)", x_range=selected_countries, y_axis_label='Inflation Rate (%)',
               width=900, height=400, tools="pan,wheel_zoom,reset")
    p.circle(x=jitter('country_name', width=0.6, range=p.x_range), y='inflation_rate', source=ColumnDataSource(data), alpha=0.4, size=8)
    return p
def make_box_plot(selected_countries):
    from bokeh.transform import jitter
    data = df_filtered[df_filtered['country_name'].isin(selected_countries)]
    p = figure(title="Inflation Rate Distribution (Box Plot Approx.)", x_range=selected_countries, y_axis_label='Inflation Rate (%)',
                width=900, height=400, tools="pan,wheel_zoom,reset")
    p.circle(x=jitter('country_name', width=0.6, range=p.x_range), y='inflation_rate', source=ColumnDataSource(data), alpha=0.4, size=8)
    return p
def make_average_trend_chart(selected_countries):
    data = df_filtered[df_filtered['country_name'].isin(selected_countries)]
    
    # Hitung rata-rata per tahun
    avg_by_year = data.groupby('year')['inflation_rate'].mean().reset_index()
    avg_by_year.columns = ['year', 'average_inflation']
    
    source = ColumnDataSource(avg_by_year)
    
    p = figure(title="Average Inflation Trend (Line Chart)", 
                x_axis_label="Year", y_axis_label="Average Inflation Rate (%)", 
                width=900, height=400, tools="pan,wheel_zoom,reset,hover")
    
    p.line(x='year', y='average_inflation', source=source, line_width=3, color='green', legend_label="Avg Inflation")
    p.circle(x='year', y='average_inflation', source=source, size=6, color='green')
    
    p.legend.location = "top_left"
    p.add_tools(HoverTool(
        tooltips=[("Year", "@year"), ("Avg Inflation", "@average_inflation{0.00}%")]
    ))
    
    return p


# --- MAIN AREA ---
st.markdown("---")
if chart_type == "Line Chart":
    st.subheader("Line Chart: Forecast for a Single Country")
    st.bokeh_chart(make_line_chart(selected_countries), use_container_width=True)
elif chart_type == "Bar Chart":
    st.subheader("Bar Chart: Maximum Inflation Rate by Country")
    st.bokeh_chart(make_bar_chart(selected_countries), use_container_width=True)
elif chart_type == "Pie Chart":
    st.subheader("Pie Chart: Average Inflation")
    st.bokeh_chart(make_pie_chart(selected_countries), use_container_width=True)
elif chart_type == "Plot Graph":
    st.subheader("Dot Plot: Inflation Over Time")
    st.bokeh_chart(make_dot_plot(selected_countries), use_container_width=True)
elif chart_type == "Stacked Area Chart":
    st.subheader("Stacked Area Chart: Inflation Composition")
    st.bokeh_chart(make_stacked_area_chart(selected_countries), use_container_width=True)
elif chart_type == "Horizontal Bar Chart":
    st.subheader("Horizontal Bar Chart: Avg Inflation")
    st.bokeh_chart(make_horizontal_bar_chart(selected_countries), use_container_width=True)
elif chart_type == "Box Plot":
    st.subheader("Box Plot: Inflation Distribution")
    st.bokeh_chart(make_box_plot(selected_countries), use_container_width=True)
elif chart_type == "Average Trend Line":
    st.subheader("Line Chart: Average Trend")
    st.bokeh_chart(make_average_trend_chart(selected_countries), use_container_width=True)

else:
    st.subheader("Multi-Line Chart: Compare Multiple Countries")
    st.bokeh_chart(make_multi_line_chart(selected_countries), use_container_width=True)

st.markdown("""
---
*Made with ❤️ using Streamlit and Bokeh*
""") 