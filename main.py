
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

def is_date(string):
    try:
        pd.to_datetime(string, errors='raise')
        return True
    except ValueError:
        return False

# Number of entries per screen
# N = 15
fileUpload = st.file_uploader("Upload a Dataset", type=["csv", "txt"])
# with st.sidebar:
# 	selected = option_menu("", ["Explore dataset","Pre process","Analysis","Dashboard"])
selected = option_menu("", ["Explore dataset","Analysis","Dashboard"])
if fileUpload is not None:
    df = pd.read_csv(fileUpload,encoding='latin')
    if 'df' not in st.session_state:
        st.session_state.df = df
        for col in st.session_state.df.columns:
            numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

            if st.session_state.df[col].dtype == 'object' and all(st.session_state.df[col].apply(is_date)):
                print('converted to date format',col)
                st.session_state.df[col] = pd.to_datetime(df[col])

    
            if st.session_state.df[col].dtype in numerics and st.session_state.df[col].nunique() <= 31:
                print('converted to object',col,st.session_state.df[col].nunique())
                st.session_state.df[col]=st.session_state.df[col].apply(str)

        duplicate = st.session_state.df.duplicated().tolist()
        if True in duplicate:
            dup =  st.session_state.df[st.session_state.df.duplicated()]
            st.session_state.df = df.drop_duplicates()

    if selected == "Explore dataset" and fileUpload is not None:
        rows = st.session_state.df.shape[0]  
        cols = st.session_state.df.shape[1] 
        st.write("Number of rows : ",rows)
        st.write("Number of columns : ",cols)
        st.dataframe(filter_dataframe(st.session_state.df))

        # if st.button('preprocess'):

     

    # if selected == "Pre process":
    #     # if 'isDuplicate' not in st.session_state:
    #     #     st.session_state.isDuplicate = 'No'

     
    #     for col in st.session_state.df.columns:
    #         data["columns"].append(col)
    #         data["Datatype"].append(st.session_state.df[col].dtype)
    #         data["missing counts"].append(st.session_state.df[col].isnull().sum())
    #         data["unique"].append(st.session_state.df[col].nunique())
                
              
            
    #     details = pd.DataFrame(data)
    #     test =details.astype(str)
    #     st.dataframe(test)
    #     duplicate = st.session_state.df.duplicated().tolist()
    #     if True in duplicate:
    #         # st.title("Duplicate rows")
    #         dup =  st.session_state.df[st.session_state.df.duplicated()]
    #         # st.dataframe(dup)
    #         st.session_state.df = df.drop_duplicates()
            # if st.button('Remove Duplicate rows'):
            #     st.session_state.isDuplicate = 'Yes'
                
            #     st.write('duplicate rows removed')
            #     st.write(len(st.session_state.df))
                
        # else:
        #     st.title("No duplicates")

        # st.title("Data type conversion")

        # dict = {}
        
        # for col in st.session_state.df.columns:
            
        #     st.write("Actual data type",st.session_state.df[col].dtype)
        #     val = st.radio(col,('none','object','Datetime','remove',))
           
        #     dict[col] =  val
        
        # if st.button('process'):
        #     st.write(dict)
        #     for key,value in dict.items():
        #         if value == 'object':
        #             st.session_state.df[key]=df[key].apply(str)
        #             st.write('converted to object(str)')
        #         elif value == 'Datetime':
        #             st.session_state.df[key] = pd.to_datetime(df[key])
        #             st.write('converted to Datetime format')
        #         elif value == 'remove':
        #             # st.session_state.df = df.drop(columns=[key])
        #             st.session_state.df = st.session_state.df.drop(columns=[key])
        #             st.write('column is removed')
        #         elif value == 'none':
        #             pass
           
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
            
           
            df = st.session_state.df
            with st.sidebar:

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
                    # to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
                    obj_column = list(df.select_dtypes(['object']).columns)
                    for column in obj_column:
                        left, right = st.columns((1, 20))
                        # Treat columns with < 10 unique values as categorical
                        if df[column].dtype == 'object' and df[column].dtype != 'int64':
                            user_cat_input = right.multiselect(
                                f"{column}",
                                df[column].unique(),
                                default=list(df[column].unique()),
                            )
                            df = df[df[column].isin(user_cat_input)]
                        # elif is_numeric_dtype(df[column]):
                        #     _min = float(df[column].min())
                        #     _max = float(df[column].max())
                        #     step = (_max - _min) / 100
                        #     user_num_input = right.slider(
                        #         f"Values for {column}",
                        #         min_value=_min,
                        #         max_value=_max,
                        #         value=(_min, _max),
                        #         step=step,
                        #     )
                        #     df = df[df[column].between(*user_num_input)]
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
                        # else:
                        #     user_text_input = right.text_input(
                        #         f"Substring or regex in {column}",
                        #     )
                        #     if user_text_input:
                        #         df = df[df[column].astype(str).str.contains(user_text_input)]
            
            # st.dataframe(df)
            # col = st.columns((2, 4.5, 2), gap='medium')
                                
            # with col[0]:
            #     for column in st.session_state.df.columns:
            #         if is_numeric_dtype(st.session_state.df[column]) and st.session_state.df[column].nunique() > 10:
            #             st.text(column)
            #             st.text("%.2f" % st.session_state.df[column].mean())
            # with col[1]:
            #     for column in st.session_state.df.columns:
            #         if is_categorical_dtype(st.session_state.df[column]) or st.session_state.df[column].nunique() < 10:
            #             unique_values_counts = st.session_state.df[column].value_counts().reset_index()
            #             unique_values_counts.columns = [column, 'Count']
            #             fig = px.bar(unique_values_counts, x=column, y='Count', title=f'Bar Graph of Unique Values in {column}')
            #             st.plotly_chart(fig, theme=None, use_container_width=True)
                                
            ColumnOption = st.multiselect("Select the column name to analyze",df.columns)
            Catcol =  []
            for col in df.columns:
                if df[col].dtype == 'object':
                    Catcol.append(col)
            for col in ColumnOption:
                # print(is_categorical_dtype(st.session_state.df[col]), col)
                
                if is_numeric_dtype(df[col]) and df[col].nunique() > 10:
                            ColumnOption = st.radio("Select the column name to analyze",Catcol) 
                        #    var =  col+ColumnOption
                        #    print('---------------------------------------')
                        #    print(ColumnOption, col,var)
                            var = df.groupby(ColumnOption)[col].sum().reset_index()
                            print('------------------')
                            print(var)
                            fig2 = px.line(var, x=ColumnOption, y=col, title='Life expectancy in Canada')
                            st.plotly_chart(fig2, theme=None, use_container_width=True)
                            st.text(col)
                            st.text("%.2f" % df[col].mean())
                if (df[col].dtype == 'object' or df[col].nunique() < 10):
                            print('---------------------entered---------------')
                            unique_values_counts = df[col].value_counts().reset_index()
                            print(col)
                            unique_values_counts.columns = [col, 'Count']
                            fig1 = px.bar(unique_values_counts, x=col, y='Count', title=f'Bar Graph of Unique Values in {col}')
                            st.plotly_chart(fig1, theme=None, use_container_width=True)
            
               






                

