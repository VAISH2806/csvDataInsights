
import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
#from streamlit_card import card


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df

# Number of entries per screen
# N = 15
fileUpload = st.file_uploader("Upload a Dataset", type=["csv", "txt"])
with st.sidebar:
	selected = option_menu("", ["Explore dataset","Pre process","Analysis","Dashboard"])
if fileUpload is not None:
    df = pd.read_csv(fileUpload,encoding='latin')
    if 'df' not in st.session_state:
        st.session_state.df = df
    if selected == "Explore dataset" and fileUpload is not None:
        rows = st.session_state.df.shape[0]  
        cols = st.session_state.df.shape[1] 
        st.write("Number of rows : ",rows)
        st.write("Number of columns : ",cols)
        st.dataframe(filter_dataframe(st.session_state.df))

     

    if selected == "Pre process":
        if 'isDuplicate' not in st.session_state:
            st.session_state.isDuplicate = 'No'

        st.title("Analyzing the Data")
        data = {"columns":[],"Datatype":[],"missing counts":[],"unique":[]}
        for col in st.session_state.df.columns:
            data["columns"].append(col)
            data["Datatype"].append(st.session_state.df[col].dtype)
            data["missing counts"].append(st.session_state.df[col].isnull().sum())
            data["unique"].append(st.session_state.df[col].nunique())
            
        details = pd.DataFrame(data)
        test =details.astype(str)
        st.dataframe(test)
        duplicate = st.session_state.df.duplicated().tolist()
        if True in duplicate:
            st.title("Duplicate rows")
            dup =  st.session_state.df[st.session_state.df.duplicated()]
            st.dataframe(dup)
            if st.button('Remove Duplicate rows'):
                st.session_state.isDuplicate = 'Yes'
                st.session_state.df = df.drop_duplicates()
                st.write('duplicate rows removed')
                st.write(len(st.session_state.df))
                
        else:
            st.title("No duplicates")

        st.title("Data type conversion")

        dict = {}
        
        for col in df.columns:
            
            st.write("Actual data type",df[col].dtype)
            val = st.radio(col,('none','object','Datetime','remove',))
           
            dict[col] =  val
        
        if st.button('process'):
            st.write(dict)
            for key,value in dict.items():
                if value == 'object':
                    st.session_state.df[key]=df[key].apply(str)
                    st.write('converted to object(str)')
                elif value == 'Datetime':
                    st.session_state.df[key] = pd.to_datetime(df[key])
                    st.write('converted to Datetime format')
                elif value == 'remove':
                    st.session_state.df = df.drop(columns=[key])
                    st.write('column is removed')
                elif value == 'none':
                    pass
           
    if selected == "Analysis":
        analyze = option_menu("Uni-variate Analysis", ["Categorical","Numerical"])
        if analyze == "Categorical":
            obj_column = list(st.session_state.df.select_dtypes(['object']).columns)
            for col in obj_column:
                # Create a DataFrame with counts of unique values in the specified column
                unique_values_counts = st.session_state.df[col].value_counts().reset_index()
                unique_values_counts.columns = [col, 'Count']

                # Create a bar graph using Plotly Express
                fig = px.bar(unique_values_counts, x=col, y='Count', title=f'Bar Graph of Unique Values in {col}')

                # Show the plot
                st.plotly_chart(fig, theme=None, use_container_width=True)
        if analyze == "Numerical":
            for col in st.session_state.df.columns:
                if st.session_state.df[col].dtype != 'object':
                    val = st.radio(col,('Histogram','Boxplot',))
                    if val == "Histogram":
                        fig = px.histogram(st.session_state.df, x=col)
                        st.plotly_chart(fig, theme=None, use_container_width=True)
                    if val == "Boxplot":
                        fig = go.Figure(data=[go.Box(x=st.session_state.df[col])])
                        st.plotly_chart(fig, theme=None, use_container_width=True)

    if selected == "Dashboard":
        data = st.session_state.df
        obj_column = list(df.select_dtypes(['object']).columns)
        selected = []
        dict = {}
        data = df
        index = 1
        cname = None
        st.sidebar.title("Attributes")
        for col in obj_column:
            unique_values_set = set(df[col])
            unique_values = tuple(unique_values_set)
            unique_values = ('all',) + unique_values 
            # cname = col+str(index)
            # cname =  st.radio(col,unique_values)
            dict[col] =  st.sidebar.radio(col,unique_values)
            index+=1
            # print(cname)
        
        actions = st.radio('actions',('reset','filter',))
        
        if actions == 'filter':

            final_condition = None
            # print('---------------------------------------')
            # print(dict)
            for key,val in dict.items():
                if dict[key] != 'all':
                    condition = (df[key] == val)
                    # st.dataframe(condition)
                    # print('----------key--------------')
                    # print(key)
                    if final_condition is None:
                        final_condition = condition
                    else:
                        final_condition = final_condition & condition
                # elif dict[key] == 'all':
                # 	del dict[key]
            
            
            
            data = df[final_condition]
            names = []
            # for col in data.columns:
            # 	if data[col].dtype != 'object':
            # 		names.append(col)
                
            # names_c = names
            # names_c = st.columns(len(names))
            # num_dict = {}
            
            # for  i in range(len(names)):
            # 	# names_c[i].metric(names[i],"%.2f" % data[names[i]].mean())
            # 	num_dict[names[i]] =  "%.2f" % data[names[i]].mean()
            
                
            # metric_row(num_dict)

            nam = []
            for col in df.columns:
                if df[col].dtype != 'object':
                    nam.append(col)
            
            s = len(nam)
            # names_c = names
            index = 0
            names = nam
            while(index<s): 
                names_c = names[index:index+3]
                nam =  names[index:index+3]   
                print('----------------------------------',names[index:index+3])
                names_c = st.columns(3)
                for  i in range(3):
                    names_c[i].metric(nam[i],"%.2f" % df[nam[i]].mean())
                
                index+=3
            
                # style_metric_cards(border_left_color="#DBF227")
                    
                    
                # style_metric_cards(border_left_color="#DBF227")
            # st.dataframe(subcat)
            # for key,values in dict.items():
            # 	if dict[key] != 'all':
            # 		print(key,values)
            # 		data = data[data[key]==values]
            # 		break
            
            # st.dataframe(data)
            obj_column = list(data.select_dtypes(['object']).columns)
            for col in obj_column:
                # Create a DataFrame with counts of unique values in the specified column
                unique_values_counts = data[col].value_counts().reset_index()
                unique_values_counts.columns = [col, 'Count']

                # Create a bar graph using Plotly Express
                fig = px.bar(unique_values_counts, x=col, y='Count', title=f'Bar Graph of Unique Values in {col}')

                # Show the plot
                st.plotly_chart(fig, theme=None, use_container_width=True)
            for col in data.columns:
                if data[col].dtype != 'object':
                    fig = px.histogram(data, x=col)
                    st.plotly_chart(fig, theme=None, use_container_width=True)

        elif actions == 'reset':
            # print(dict)
            for key,val in dict.items():
                dict[key] = 'all'
            # print(dict)
