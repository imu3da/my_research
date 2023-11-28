# -------------------------
# インポート
# -------------------------
import pandas as pd
import requests
import time
import unicodedata
from bs4 import BeautifulSoup
from urllib.parse import urlparse


# -------------------------
# 関数
# -------------------------
# データフレームをマークダウン記法の箇条書きに変換する関数
def format_df_to_md(df):
    markdown_text = ''
    for _, row in df.iterrows():
        markdown_text += f"- {row.iloc[0]}: {row.iloc[1]}\n"
    return unicodedata.normalize('NFKC', markdown_text.rstrip())

# タプルをマークダウン記法の番号付き箇条書きに変換する関数
def format_tuple_list_to_md(tuple_list):
    markdown_text = ''
    for item in tuple_list:
        markdown_text += f"{item[0]}. {item[1]}\n"
    return unicodedata.normalize('NFKC', markdown_text.rstrip())

# 各カテゴリーの投稿データを取得する関数
def get_post(category_url):
    page = 1
    post_data = pd.DataFrame()
    base_url = urlparse(category_url).scheme + "://" + urlparse(category_url).netloc
    while True:
        page_url = category_url + '?page=' + str(page)
        response = requests.get(page_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        post_title_div = soup.find_all('div', class_='post-title')
        if not post_title_div:
            break
        for post in post_title_div:
            post_title = post.find('a').text.strip()
            post_title = unicodedata.normalize('NFKC', post_title)
            post_url = post.find('a')['href']
            row_df = pd.DataFrame({'post_title': [post_title], 'post_url': [base_url+post_url]})
            post_data = pd.concat([post_data, row_df], ignore_index=True)
        page += 1
    return post_data

# post-contentクラスの内容を取得する関数
def get_post_content(soup):
    post_content_data = ''
    post_content_div = soup.find('div', class_='post-content')
    if post_content_div:
        post_contents = post_content_div.find_all('p')
        post_content_data = '\n'.join(p.text.strip() for p in post_contents).rstrip()
    return post_content_data

# post-tableクラスの内容を取得する関数
def get_post_info_table(soup):
    post_info_data_list = []
    post_info_tables = soup.find_all('table', class_='post-info-table')
    for post_info_table in post_info_tables:
        rows = post_info_table.find_all('tr')
        for row in rows:
            header = row.find('th').text.strip()
            data = row.find('td').text.strip()
            post_info_data_list.append({'th': header, 'td': data})
    post_info_data = pd.DataFrame(post_info_data_list)
    return format_df_to_md(post_info_data)

# slider-blockクラスの内容を取得する関数
def get_slider_block(soup):
    slider_block_data = {}
    slider_blocks = soup.find_all('div', class_='slider-block')
    for slider_block in slider_blocks:
        slider_block_title = slider_block.find('div', class_='slider-block-title').text.strip()
        post_titles = slider_block.find_all('div', class_='post-title')
        post_titles_list = []
        for post in post_titles:
            a_tag = post.find('a')
            if a_tag:
                post_titles_list.append(a_tag.text.strip())
            else:
                post_titles_list.append(post.text.strip())
        if slider_block_title in slider_block_data:
            slider_block_data[slider_block_title].extend(post_titles_list)
        else:
            slider_block_data[slider_block_title] = post_titles_list
    slider_block_data = {k: ', '.join(v) for k, v in slider_block_data.items()}
    slider_block_data_df = pd.DataFrame(list(slider_block_data.items()), columns=['slider_block_title', 'post_title'])
    return format_df_to_md(slider_block_data_df)

# other_blockクラスの内容を取得する関数
def get_other_block(soup):
    other_block_data = {}
    other_blocks = soup.find_all('div', class_='other-block')
    for other_block in other_blocks:
        other_block_title = other_block.find('div', class_='other-block-title').text.strip()
        other_block_contents = other_block.find_all('a')
        other_block_contents_list = [content.text.strip() for content in other_block_contents]
        if other_block_title in other_block_data:
            other_block_data[other_block_title].extend(other_block_contents_list)
        else:
            other_block_data[other_block_title] = other_block_contents_list
    other_block_data = {k: ', '.join(v) for k, v in other_block_data.items()}
    other_block_data_df = pd.DataFrame(list(other_block_data.items()), columns=['other_block_title', 'other_block_content'])
    return format_df_to_md(other_block_data_df)

# index-list js-indexBox-content page-link-scrollクラスの内容を取得する関数
def get_index_box_content(soup):
    index_box_content_data = []
    index_box_content_data_md = ''
    index_box_contents = soup.find('ul', class_='index-list js-indexBox-content page-link-scroll')
    if index_box_contents:
        index_box_content_items = index_box_contents.find_all('li')
        for item in index_box_content_items:
            index_number = item.find('span', class_='index-number').text.strip()
            index_text = item.find('a', class_='index-text').text.strip()
            index_box_content_data.append((index_number, index_text))
        index_box_content_data_md = format_tuple_list_to_md(index_box_content_data)
    return index_box_content_data_md

# 投稿の基本的な中身をまとめて取得する関数
def get_post_description(soup):
    post_content = get_post_content(soup)
    post_info_table = get_post_info_table(soup)
    slider_block = get_slider_block(soup)
    other_block = get_other_block(soup)
    post_description = '\n\n'.join([post_content, post_info_table, slider_block, other_block]).strip()
    return post_description

# カテゴリーごとに処理を行う関数
def process_scraping(categories, user_input):
    category_name = categories[user_input]['name']
    category_url = base_url + categories[user_input]['path']
    post_data = get_post(category_url)
    post_data_list = []
    post_count = len(post_data)
    for index, row in post_data.iterrows():
        post_title = unicodedata.normalize('NFKC', row['post_title'])
        post_url = row['post_url']
        response = requests.get(post_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        if category_name == 'モデルコース':
            post_title_course = '「' + post_title + '」コース'
            post_description = get_index_box_content(soup)
            post_data_list.append((post_title_course, post_url, post_description))
            print(f'「{post_title}」のデータを取得しました。 ({index+1}/{post_count})')
            time.sleep(2)
        else:
            post_description = get_post_description(soup)
            post_data_list.append((post_title, post_url, post_description))
            print(f'「{post_title}」のデータを取得しました。 ({index+1}/{post_count})')
            time.sleep(2)
    df = pd.DataFrame(post_data_list, columns=['name', 'url', 'description'])
    return df


# -------------------------
# 変数の設定
# -------------------------
base_url = 'https://www.hida-kankou.jp/'
categories = {
    1: {'name': 'モデルコース', 'path': 'courses'},
    2: {'name': 'スポット', 'path': 'spot'},
    3: {'name': '体験', 'path': 'plan'},
    4: {'name': 'イベント', 'path': 'event'},
    5: {'name': 'グルメ・おみやげ', 'path': 'product'},
    6: {'name': '宿泊予約', 'path': 'reserve'},
    9: {'name': '終了', 'path': 'none'}
}
output_dir = 'my_research/output/'


# -------------------------
# 実際の処理
# -------------------------
print('--------------------')
for key, value in categories.items():
    print(f"{key}: {value['name']}")
print('--------------------')
try:
    user_input = int(input('このスクリプトは、飛騨市公式観光サイト「飛騨の旅」からデータを取得するものです。データを取得したいカテゴリーの数字を入力してください。: '))
    print('--------------------')
    if user_input not in categories:
        print('入力された数字のカテゴリーは存在しません。スクリプトを終了します。')
        exit()
    elif  categories[user_input]['name'] == '終了':
        print('スクリプトを終了します。')
        exit()
    else:
        print('データの取得を開始します。')
        df = process_scraping(categories, user_input)
        df.to_csv(f'{output_dir}{categories[user_input]['path']}_from_hidanotabi.csv', index=False)
        print('データの取得が完了しました。')
        print('--------------------')
except ValueError:
    print('数字以外が入力されました。スクリプトを終了します。')
    exit()
