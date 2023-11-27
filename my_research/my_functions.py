# -------------------------
# インポート
# -------------------------
import ast, boto3, os, random
from decimal import Decimal
from openai import OpenAI
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import streamlit as st


# -------------------------
# 外部との接続
# -------------------------
# 各種クライアントを用意する関数
def launch_client():
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    aws_access_key = os.environ.get('AWS_ACCESS_KEY')
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    aws_region = os.environ.get('AWS_REGION')
    client = OpenAI(api_key=openai_api_key)
    dynamodb = boto3.resource('dynamodb', region_name=aws_region, aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_access_key)
    return client, dynamodb


# -------------------------
# DynamoDB
# -------------------------
# データを読み込む関数
def read_from_dynamodb(dynamodb, table_name):
    table = dynamodb.Table(table_name)
    data = []
    response = table.scan()
    data.extend(response['Items'])
    while 'LastEvaluatedKey' in response:
        response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
        data.extend(response['Items'])
    df = pd.DataFrame(data)
    return df

# テーブルから指定のデータを削除する関数
def delete_item(dynamodb, table_name, partition_key_name, partition_key):
    table = dynamodb.Table(table_name)
    response = table.delete_item(
        Key={
            partition_key_name: partition_key
        }
    )
    return response

# テーブル内の全てのデータを削除する関数
def delete_all_items(dynamodb, table_name, partition_key_name):
    # テーブルからすべてのデータを読み込む
    df = read_from_dynamodb(dynamodb, table_name)
    # 各アイテムを削除する
    for index, row in df.iterrows():
        delete_item(dynamodb, table_name, partition_key_name, row[partition_key_name])

# データを書き込む関数
def write_to_dynamodb(dynamodb, df, table_name):
    table = dynamodb.Table(table_name)
    # DataFrame内の数値をDynamoDBが対応している型に変換
    df = df.replace(np.nan, 0)
    df = df.applymap(lambda x: Decimal(x) if isinstance(x, (int, float)) else x)
    for i in range(len(df)):
        item = df.iloc[i].to_dict()
        table.put_item(Item=item)


# -------------------------
# ベクトル
# -------------------------
# 文章をベクトル変換する関数
def get_embedding(client, text, model='text-embedding-ada-002'):
   if pd.isnull(text):
       return np.array([])
   text = text.replace('\n', ' ')
   response = client.embeddings.create(input=[text], model=model)
   return np.array(response.data[0].embedding)

# 最近傍探索
def nearest_neighbor_search(client, query, df, partition_key_name):
    query_embedded = get_embedding(client, query)
    df_name = df[[partition_key_name, 'name_embedded']].dropna()
    df_description = df[[partition_key_name, 'description_embedded']].dropna()
    df_name = df_name.rename(columns={'name_embedded': 'embedded'})
    df_description = df_description.rename(columns={'description_embedded': 'embedded'})
    # DataFrameを連結
    df_combined = pd.concat([df_name, df_description], ignore_index=True)
    # 類似度とIDを格納するリスト
    similarities_and_ids = []
    for _, row in df_combined.iterrows():
        similarity = cosine_similarity([row['embedded']], [query_embedded])[0][0]
        similarities_and_ids.append((similarity, row[partition_key_name]))
    # 類似度でソート
    similarities_and_ids.sort(reverse=True)
    # 上位2つの異なるIDを持つデータを取り出す
    top_ids = []
    for similarity, id in similarities_and_ids:
        if id not in top_ids:
            top_ids.append(id)
            if len(top_ids) == 2:
                break
    # IDに基づいてデータフレームを取得
    top_df = df[df[partition_key_name].isin(top_ids)]
    return top_df

# データフレームの内容をベクトル化する関数
def process_dataframe(df, client):
    # nameとdescriptionをベクトル化
    df['name_embedded'] = df['name'].apply(lambda x: get_embedding(client, x).tolist())
    df['description_embedded'] = df['description'].apply(lambda x: get_embedding(client, x).tolist())
    # ベクトル化されたデータを文字列に変換
    df = convert_df_to_string(df)
    return df


# -------------------------
# フォーマット
# -------------------------
# strをfloatに変換する関数
def convert_string_to_float_list(s):
    if pd.isnull(s):
        return np.nan
    else:
        return [float(i) for i in ast.literal_eval(s)]

# 文字列型のベクトルをfloatのリストに変換する関数
def convert_df_embeddings(df, columns=['name_embedded', 'description_embedded']):
    for column in columns:
        if column in df.columns:
            df[column] = df[column].apply(convert_string_to_float_list)
    return df

# floatのリストをstrに変換する関数
def convert_float_list_to_string(l):
    if l is np.nan:
        return None
    else:
        return str(l)

# floatのリストをstrに変換する関数を適用する関数
def convert_df_to_string(df, columns=['name_embedded', 'description_embedded']):
    for column in columns:
        if column in df.columns:
            df[column] = df[column].apply(convert_float_list_to_string)
    return df

# データフレームをプロンプト用にマークダウン形式に変換する関数
def df_to_markdown(df):
    markdown_str = ""
    for i, row in df.iterrows():
        if pd.notna(row['name']) and pd.notna(row['description']):
            markdown_str += f"# 「**{row['name']}**」についての追加情報\n"
            markdown_str += f"## 説明\n{row['description']}\n"
            markdown_str += "## 詳細\n"
            if pd.notna(row['address']):
                markdown_str += f"- 住所: {row['address']}\n"
            if pd.notna(row['access']):
                markdown_str += f"- アクセス: {row['access']}\n"
            if pd.notna(row['phone']):
                markdown_str += f"- 電話番号: {row['phone']}\n"
            if pd.notna(row['url']):
                markdown_str += f"- URL: [{row['name']}]({row['url']})\n"
            if pd.notna(row['strong_point_from_review']):
                markdown_str += "## 魅力\n"
                points = row['strong_point_from_review'].split(', ')
                if len(points) > 3:
                    points = random.sample(points, 3)
                for point in points:
                    markdown_str += f"- {point.strip()}\n"
            markdown_str += "\n"
    return markdown_str


# -------------------------
# その他
# -------------------------
# 質問文をもとにAIに追加する適切な情報を文字列で返す関数
def get_knowledge(client, query, df, partition_key_name='id'):
    df = nearest_neighbor_search(client, query, df, partition_key_name)
    knowledge = df_to_markdown(df)
    return knowledge

# エラーをお知らせする関数
def show_fail_toast(message=None):
    if message is not None:
        st.toast(message, icon='😱')
    else:
        messages = [
            "おっと、何かがおかしいようです。もう一度試してみてください。",
            "エラーが発生しました。でも大丈夫、これはただの気のせいのはずです。",
            "うーん、何かがうまくいかなかったようです。再度試してみてください。",
            "エラーが発生しました。でも心配しないで。",
            "不幸なことに、何かが間違っているようです。"
        ]
        st.toast(random.choice(messages), icon='😱')

# 成功をお知らせする関数
def show_success_toast(message=None):
    if message is not None:
        st.toast(message, icon='🎉')
    else:
        messages = [
            "素晴らしい！操作は成功しました。",
            "やったね！全てが順調に進んでいます。",
            "完璧！素晴らしい仕事ができました。",
            "おめでとう！操作は成功しました。",
            "すごい！やり遂げました。"
        ]
        st.toast(random.choice(messages), icon='🎉')

# 10%で風船を出す関数
def deco_balloons(probability=0.1):
    if random.random() < probability:
        st.balloons()
        show_success_toast('あ、風船が飛んできました！')

# 10%で雪を降らせる関数
def deco_snow(min_probability=0.1, max_probability=0.2):
    random_value = random.random()
    if min_probability <= random_value < max_probability:
        st.snow()
        show_success_toast('ん？雪が降り始めたみたいですね…')

# 風船と雪のどちらかをそれぞれ10%で出す関数
def deco():
    deco_balloons()
    deco_snow()

# キャッシュなどをクリアしページを更新する関数
def reload():
    st.cache_resource.clear()
    st.session_state.clear()
    st.experimental_rerun()

# ボタン状態を含めて全てを更新する関数
def all_reload(button_state):
    st.cache_resource.clear()
    st.session_state.clear()
    st.session_state[button_state] = False
    st.experimental_rerun()

# ボタンのセッションステートの初期化を行う関数
def init_session(button_state, button_label):
    if button_state not in st.session_state:
        st.session_state[button_state] = False
    if st.button(button_label):
        st.session_state[button_state] = True

# PDFからデータを読み込む関数
def read_from_pdf(uploaded_file, id):
    pdf = PdfReader(uploaded_file)
    text = ''
    for page in pdf.pages:
        text += page.extract_text()
    # ファイル名から'.pdf'拡張子を削除
    file_name, _ = os.path.splitext(uploaded_file.name)
    df = pd.DataFrame({
        'name': [file_name],
        'description': [text],
        'url': [''],
        'id': [id]
    })
    return df
