import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

def load_dataset():
  files = ["data/" + f for f in os.listdir('data') if len(f.split('.')) > 1 and f.split('.')[1] == 'csv']
  all_dataset = pd.DataFrame()
  for f in files:
    dataset = pd.read_csv(f)
    all_dataset = pd.concat([all_dataset, dataset])
  
  all_dataset = all_dataset.dropna(how='all')
  size = all_dataset.shape[0]
  all_dataset = all_dataset[:int(size/10)]
  return all_dataset

def add_user_id(dataset):
  addresses = dataset['Purchase Address'].unique()
  addresses_dict = dict()
  for i, item in enumerate(addresses):
    addresses_dict[item] = i+1

  dataset['User ID'] = dataset['Purchase Address'].map(addresses_dict)
  return dataset

def convert_col_to_int(dataset, column_name):
  dataset = dataset.loc[(dataset[column_name].str.isnumeric())]
  dataset[column_name] = pd.to_numeric(dataset[column_name])
  return dataset



dataset = load_dataset()
dataset = add_user_id(dataset)
dataset = convert_col_to_int(dataset, "Quantity Ordered")

st.title("Recommendation System with Sales Analysis Data")

st.write("Data Overview")
st.dataframe(dataset)
st.write("Total Data: " + str(dataset.shape[0]))

st.markdown("""---""")

st.write("User ID and number of unique products they've bought")
st.dataframe(dataset['User ID'].value_counts())

st.markdown("""---""")

ids = list(dataset['User ID'].unique())
ids.insert(0, "Select ID")

user_id = st.selectbox("Choose User ID", (ids))

if user_id != "Select ID":
  st.write("Selected User ID : " + str(user_id))
  customer_item_matrix = dataset.pivot_table(
      index='User ID', 
      columns='Product', 
      values='Quantity Ordered',
      aggfunc='sum'
  )

  customer_item_matrix = customer_item_matrix.applymap(lambda x: 1 if x > 0 else 0)
  user_user_sim_matrix = pd.DataFrame(
      cosine_similarity(customer_item_matrix)
  )
  user_user_sim_matrix.columns = customer_item_matrix.index
  user_user_sim_matrix['CustomerID'] = customer_item_matrix.index
  user_user_sim_matrix = user_user_sim_matrix.set_index('CustomerID')


  items_bought_by_A = set(customer_item_matrix.loc[user_id].iloc[
    customer_item_matrix.loc[user_id].to_numpy().nonzero()
  ].index)
  st.write("Items Bought by User ID : " + str(user_id) + " :")
  st.dataframe(items_bought_by_A)

  st.markdown("""---""")

  st.write("User ID with highest similiarity with User ID " + str(user_id) + " :")
  most_similiar_user = user_user_sim_matrix.loc[user_id].sort_values(ascending=False).head(10)
  st.dataframe(most_similiar_user)

  st.markdown("""---""")

  choose_b_user = st.selectbox("Select User to Recommend : ", (most_similiar_user.index))

  if choose_b_user != user_id:
    items_bought_by_B = set(customer_item_matrix.loc[choose_b_user].iloc[
      customer_item_matrix.loc[choose_b_user].to_numpy().nonzero()
    ].index)

    st.write("Items Bought by User ID : " + str(choose_b_user) + " :")
    st.dataframe(items_bought_by_B)

    st.markdown("""---""")

    items_to_recommend_to_B = items_bought_by_A - items_bought_by_B
    recommended = dataset.loc[
        dataset['Product'].isin(items_to_recommend_to_B), 
        ['Product']
    ].drop_duplicates().set_index('Product')
    st.write("Items that will be recommended to User ID " + str(choose_b_user) + " is/are : ")
    st.dataframe(items_to_recommend_to_B)





