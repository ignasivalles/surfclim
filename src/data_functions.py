import numpy as np
import pandas as pd
import holoviews as hv
from scipy.signal import savgol_filter
from bokeh.models import HoverTool
hv.extension('bokeh')

def detect_water_temp(temp, time, window_size=6, only_water_val=True):

    # Classify based on moving variance
    moving_avg = pd.Series(temp).rolling(window=window_size, min_periods=1).mean().to_numpy()
    moving_var = pd.Series(temp).rolling(window=window_size, min_periods=1).var().to_numpy()

    var_threshold = np.nanmedian(moving_var)*4
    mean_threshold = np.nanmedian(moving_avg)
    print(moving_var)

    ithreshold = np.where(moving_var > var_threshold)[0][0]
    #print(ithreshold)

    state = np.full(temp.shape, 'air')
    state[ithreshold:] = np.where((moving_var[ithreshold:] < var_threshold) ,'wat','air')
    #print(state)

    #state = np.where((moving_var < var_threshold) & (moving_avg < mean_threshold), 'water', 'air')

    # Forward fill and backward to smooth
    state_filled = pd.Series(state).ffill().bfill().to_numpy()
    water_data = temp[state_filled == 'wat']
    air_data = temp[state_filled == 'air']
    water_time = time[state_filled == 'wat']
    air_time = time[state_filled == 'air']

    if only_water_val:
      return np.nanmean(water_data)
    else:
      return water_data, water_time, moving_var, var_threshold


import pandas as pd
import numpy as np

def detect_water_temp2(temp, time, window_size=6, only_water_val=True):
    def calculate_state(temp, moving_var, var_threshold):
        ithreshold = np.where(moving_var > var_threshold)[0][0] if np.any(moving_var > var_threshold) else None
        state = np.full(temp.shape, 'air')
        if ithreshold is not None:
            state[ithreshold:] = np.where((moving_var[ithreshold:] < var_threshold), 'wat', 'air')
        return state

    # Classify based on moving variance
    moving_avg = pd.Series(temp).rolling(window=window_size, min_periods=1).mean().to_numpy()
    moving_var = pd.Series(temp).rolling(window=window_size, min_periods=1).var().to_numpy()

    var_threshold = np.nanmedian(moving_var) * 4
    state = calculate_state(temp, moving_var, var_threshold)

    # Forward fill and backward to smooth
    state_filled = pd.Series(state).ffill().bfill().to_numpy()
    water_data = temp[state_filled == 'wat']

    # Check if water_data is empty and adjust var_threshold if necessary
    if len(water_data) == 0:
        var_threshold /= 4
        state = calculate_state(temp, moving_var, var_threshold)
        state_filled = pd.Series(state).ffill().bfill().to_numpy()
        water_data = temp[state_filled == 'wat']

    if only_water_val:
        return np.nanmean(water_data)
    else:
        water_time = time[state_filled == 'wat']
        air_data = temp[state_filled == 'air']
        air_time = time[state_filled == 'air']
        return water_data, water_time, moving_var, var_threshold

def detect_water_temp_v2(temp, time, window_size=4, only_water_val=True):
    # Calculate the rolling mean and rolling variance
    moving_avg = pd.Series(temp).rolling(window=window_size, min_periods=1).mean().to_numpy()
    moving_var = pd.Series(temp).rolling(window=window_size, min_periods=1).var().to_numpy()

    # Define a threshold for detecting similar values, using a small variance threshold
    var_threshold = np.nanmean(moving_var) * 0.2

    # Identify periods with low variance
    similar_periods = moving_var < var_threshold

    # Find the longest period of similar values
    max_length = 0
    max_start = 0
    current_start = 0
    current_length = 0

    for i in range(len(similar_periods)):
        if similar_periods[i]:
            if current_length == 0:
                current_start = i
            current_length += 1
        else:
            if current_length > max_length:
                max_length = current_length
                max_start = current_start
            current_length = 0

    if current_length > max_length:
        max_length = current_length
        max_start = current_start

    longest_similar_period = temp[max_start:max_start + max_length]
    time_similar_period = time[max_start:max_start + max_length]

    if only_water_val:
        return np.nanmean(longest_similar_period), time_similar_period[0]
    else:
        water_time = time[max_start:max_start + max_length]
        return longest_similar_period, water_time, moving_var, var_threshold

def datetime_to_decimal_year(date):
    year = date.astype('datetime64[Y]').astype(int) + 1970
    start_of_year = np.datetime64(f'{year}-01-01')
    end_of_year = np.datetime64(f'{year + 1}-01-01')
    year_duration = (end_of_year - start_of_year).astype('timedelta64[D]').astype(int)
    date_position = (date - start_of_year).astype('timedelta64[D]').astype(int)
    decimal_year = date_position / year_duration * 12  # Scale to 0-12 where January=0 and December=11
    return decimal_year



def get_data_from_temp_sensors(filepath, team_name='raw', lat= None, lon= None ):

    data=extract_lat_lon_temp_time(filepath)
    time=data.time.values
    temp=data.temp.values

    water_temp, water_time = detect_water_temp_v2(temp, time)
    time_str = np.datetime_as_string(water_time, unit='h')
    fractional_time=datetime_to_decimal_year(water_time)
    
    if not lat:
      lat=data.lat.values[0]
      lon=data.lon.values[0]

    df = pd.DataFrame([{  # Approx. week 52 for Dec 24
    'Date': time_str,
    'Latitude': lat,
    'Longitude': lon,
    'Temperature': water_temp,
    'fractional_time': fractional_time,
    'Team': team_name}])

    return df

def extract_lat_lon_temp_time(file_path):
    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Extract latitude and longitude from metadata
    lat, long = None, None
    for line in lines:
        if line.startswith('lat'):
            lat = float(line.split(',')[1].strip())
        if line.startswith('long'):
            long = float(line.split(',')[1].strip())

    # Find the line index where actual data starts
    data_start_idx = next((idx for idx, line in enumerate(lines) if line.strip().startswith('time,temp')), None)

    # Load the data into a DataFrame
    data = pd.read_csv(file_path, skiprows=data_start_idx)

    # Convert 'time' to datetime and handle missing data
    data['time'] = pd.to_datetime(data['time'], errors='coerce')
    data['temp'] = pd.to_numeric(data['temp'], errors='coerce')

    # Add latitude and longitude to the DataFrame
    data['lat'] = lat
    data['lon'] = long

    return data[['time', 'temp', 'lat', 'lon']]


def plot_climatological_year(file_path):
    
    # Load the dataset
    df = pd.read_excel(file_path, header=0)
    df = df.rename(columns={'año': 'year', 'mes': 'month', 'dia': 'day', 'temperatura agua': 'temperatura'})

    # Create a Date column
    df['Date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df['Date'] = df['Date'].astype('datetime64[ns]')

    # Group by month for climatological statistics
    grouped = df.groupby('month')['temperatura']
    q1 = grouped.quantile(0.25)
    q3 = grouped.quantile(0.75)
    iqr = q3 - q1
    lower_whisker = q1 - 1.5 * iqr
    upper_whisker = q3 + 1.5 * iqr
    medians = grouped.median()


    # Apply the function to calculate fractional year
    print(df['Date'])
    df['fractional_time'] = [datetime_to_decimal_year(date) for date in df['Date'].values]


    # Weekly indices for interpolation
    weekly_indices = np.linspace(0, 12, 84)
    weeks_per_month = [4.345] * 12
    cumulative_weeks = np.cumsum([0] + weeks_per_month)

    # Interpolate whiskers and medians to weekly scale
    print(len(df['fractional_time']))
    print(len(medians))
    weekly_medians = np.interp(weekly_indices, np.arange(0,12,1), medians)
    weekly_lower = np.interp(weekly_indices, np.arange(0,12,1), lower_whisker)
    weekly_upper = np.interp(weekly_indices, np.arange(0,12,1), upper_whisker)

    # Smooth whiskers and medians
    smoothed_medians = savgol_filter(weekly_medians, 7, 2)
    smoothed_lower = savgol_filter(weekly_lower, 7, 2)
    smoothed_upper = savgol_filter(weekly_upper, 7, 2)

    # Create the climatological year plot using Holoviews (with vdims)
    climato_plot = hv.Area(
        (weekly_indices, smoothed_lower, smoothed_upper),
        vdims=['y', 'y2'],
        label="IQR-based Range"
    ).opts(
        color='lightblue',
        alpha=0.5,
        title='Climatological Year Temperature Range',
        xlabel='Month',
        ylabel=r"Temperature [°C]",
        responsive=True,
        )

    # Add the smoothed median line (in weekly scale)
    climato_plot_median = hv.Curve(
        (weekly_indices, smoothed_medians),
        label='Median'
    ).opts(
        color='blue',
        line_width=2,
        responsive=True,
        )

    month_boundaries = [
      0,    # January
      1, # February
      2,  # March
      3, # April
      4,  # May
      5, # June
      6,  # July
      7, # August
      8,  # September
      9, # October
      10,  # November
      11  # December
    ]

    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Create the combined plot
    combined_plot = climato_plot * climato_plot_median

# Set x-ticks and labels for months
    combined_plot.opts(
      xticks=[(i, label) for i, label in zip(month_boundaries, month_labels)],
      show_grid=True
    )

    # Show the combined plot
    return combined_plot

def plot_obs(df):
    hover = HoverTool(tooltips=[
        ('Date', '@Date{%F}'),
        ('Temperature', '@Temperature'),
        ('Anomaly', '@Temperature_Anomaly')
    ], formatters={'@Date': 'datetime'})

    scatter_plot = hv.Scatter(
        df,
        kdims=['fractional_time'],
        vdims=['Temperature', 'Temperature_Anomaly', 'Date']
    ).opts(
        tools=[hover, 'pan', 'wheel_zoom', 'box_zoom', 'reset'],
        color='Temperature_Anomaly',
        cmap='coolwarm',
        clim=(-3, 3),
        size=5,
        alpha=0.4,
        xlabel='Month',
        ylabel=r"Temperature [°C]",
        legend_position='top_left'
    )
    return scatter_plot


def plot_rolling_by_year(df, window=11, min_year=2025):
    """
    For each year >= min_year, compute a rolling median + Savitzky-Golay smooth
    of temperature vs fractional_time and return a HoloViews Overlay of Curves.
    Colors cycle automatically so future years are handled.
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date', 'Temperature', 'fractional_time'])
    df['year'] = df['Date'].dt.year
    df = df[df['year'] >= min_year]

    # Distinct warm colours — avoid blue already used by the climatology median
    year_colors = ['#e74c3c', '#2ecc71', '#f39c12', '#9b59b6',
                   '#e67e22', '#1abc9c', '#e91e63', '#ff5722']

    curves = []
    for i, year in enumerate(sorted(df['year'].unique())):
        year_df = (df[df['year'] == year]
                   .sort_values('fractional_time')
                   .reset_index(drop=True))
        if len(year_df) < 5:
            continue

        w = min(window, max(3, len(year_df) // 2))
        rolling = (year_df['Temperature']
                   .rolling(window=w, min_periods=3, center=True)
                   .median())
        valid = rolling.notna()
        if valid.sum() < 5:
            continue

        # Apply Savitzky-Golay for additional smoothness
        vals = rolling[valid].values
        sg_window = min(len(vals), 15)
        if sg_window % 2 == 0:
            sg_window -= 1
        sg_window = max(sg_window, 5)
        smoothed = savgol_filter(vals, window_length=sg_window, polyorder=2)

        color = year_colors[i % len(year_colors)]
        curve = hv.Curve(
            (year_df['fractional_time'][valid].values, smoothed),
            label=str(year)
        ).opts(color=color, line_width=2.5)
        curves.append(curve)

    return hv.Overlay(curves) if curves else hv.Curve([])

def get_data_from_temp_sensors_full(filepath, team_name='Escuela Salle', location='IEO', lat=None, lon=None):
    # Read the data
    data = pd.read_csv(filepath)
    
    # Extract time and temperature data
    time = pd.to_datetime(data.iloc[21::].iloc[:, 0]).to_numpy()
    temp = data.iloc[21::].iloc[:, 1].to_numpy().astype('float')
    
    # Process the data
    water_data, water_time, moving_var, var_threshold = detect_water_temp(
        temp, time, window_size=6, only_water_val=False
    )
    
    # Convert times to string format with minute precision
    time_str = np.array([t.astype('datetime64[m]').astype(str) for t in water_time])
    
    # Get latitude and longitude if not provided
    if not lat:
        lat = float(data.iloc[16][1])
        lon = float(data.iloc[17][1])
    
    # Create the DataFrame in "tidy" format
    df_full = pd.DataFrame({
        'Temperature': water_data,       # Expand temperatures
        'Date': water_time,                # Expand corresponding times
        'Latitude': [lat] * len(water_data),    # Repeat metadata for each row
        'Longitude': [lon] * len(water_data),
        'Team': [team_name] * len(water_data),
        'Location': [location] * len(water_data),
    })

    return df_full


