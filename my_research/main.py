# -------------------------
# インポート
# -------------------------
import boto3, json, os, re
# from dotenv import load_dotenv
from openai import OpenAI, BadRequestError
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import my_functions as mf


# -------------------------
# デバッグ
# -------------------------
# load_dotenv()


# -------------------------
# 全体的な準備
# -------------------------
# 各種クライアントを用意する
client, dynamodb = mf.launch_client()
# 各種変数の設定
models = ['gpt-3.5-turbo-1106']
site_name = '仮名'
pages = ['❗️**AIアドバイザー** 💬', '学習データの確認・変更 🧐', '投稿文の自動生成 🤖']
forget_input = '全ての必須項目を入力してください。'
system_content = os.environ.get('AI_RULE')
table_name = os.environ.get('KNOWLEDGE_TABLE')
user_table_name = os.environ.get('USER_KNOWLEDGE_TABLE')
partition_key_name = os.environ.get('PARTITION_KEY')
# サイトの設定
st.set_page_config(page_title=site_name, page_icon='💻')
st.header(site_name)


# -------------------------
# キャッシュ
# -------------------------
@st.cache_resource
def read_from_dynamodb(_dynamodb, table_name):
    return mf.read_from_dynamodb(dynamodb, table_name)

@st.cache_resource
def convert_df_embeddings(df):
    return mf.convert_df_embeddings(df)


# -------------------------
# データの読み込み
# -------------------------
# データベースの読み込み
df = read_from_dynamodb(dynamodb, table_name)
df= df.replace(['[]', 0], np.nan).sort_values(partition_key_name)
df = convert_df_embeddings(df)
df_base = df.copy()
user_df = read_from_dynamodb(dynamodb, user_table_name)
# データフレームが空でない場合のみ、後続の処理を実行
if not user_df.empty:
    user_df = user_df.replace(['[]', 0], np.nan).sort_values(partition_key_name)
    user_df = convert_df_embeddings(user_df)
    user_df_indexed = user_df.copy()
    # dfのidの最大値を取得
    max_id = int(df[partition_key_name].max())
    # user_dfのidを更新
    user_df_indexed[partition_key_name] = range(max_id + 1, max_id + 1 + len(user_df))
    # dfとuser_dfを統合
    df = pd.concat([df, user_df_indexed], ignore_index=True, sort=False)


# -------------------------
# サイドバー
# -------------------------
page = st.sidebar.radio('メニュー', pages)
model = st.sidebar.selectbox('モデル', models, help = 'チャットと文章の自動生成に用いるAIのモデルを選択してください。(現在は1種類のみ選択できるようにしています。)')
st.session_state['model'] = model
if 'model' not in st.session_state:
    st.session_state['model'] = 'gpt-3.5-turbo-1106'
api_key_input = st.sidebar.text_input(
    'OpenAI APIキー',
    type='password',
    placeholder='skから始まるAPIキーを入力',
    help='APIキーは🔗[こちら](https://platform.openai.com/account/api-keys)のページで入手できます。(現在は入力しなくても利用できるようにしています。)'
)
if api_key_input:
    client = OpenAI(api_key = api_key_input)
answer_num = st.sidebar.slider('単位知識数', 1, 5, 3, help='これは一度にAIが利用する知識の数です。高いほど知識が豊かになりますが、別の知識と混ざり、応答内容が不正確になる可能性が上がります。')
st.session_state['answer_num'] = answer_num
history_num = st.sidebar.slider('記憶クエリ数', 1, 10, 3, help='これはAIアドバイザーとの会話での、AIが空気を読む精度のようなものです。高いほど会話全体の流れを考慮しますが、発言1回あたりの重要性が下がり、クリティカルな返答が難しくなります。')
st.session_state['history_num'] = history_num
st.sidebar.write('''
---
# <span style='color: silver; '>説明</span>
## <span style='color: gray; '>💬 AIアドバイザー</span>
AI観光アドバイザーと会話ができます。
## <span style='color: gray; '>🧐 学習データの確認・変更</span>
AIに与えている知識のデータを確認または変更できます。
## <span style='color: gray; '>🤖 投稿文の自動生成</span>
SNSやHPに掲載する文章を自動生成できます。
''', unsafe_allow_html=True)
st.sidebar.divider()
if 'reload_button_state' not in st.session_state:
    st.session_state['reload_button_state'] = False
if st.sidebar.button('再読み込み'):
    st.session_state['reload_button_state'] = True
if st.session_state['reload_button_state']:
    mf.all_reload('reload_button_state')
button_css = f'''
<style>
    div.stButton {{
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100%;
    }}
    div.stButton > button:first-child {{
        margin: auto;
    }}
</style>
'''
st.markdown(button_css, unsafe_allow_html=True)


# -------------------------
# AIアドバイザー
# -------------------------
if page == pages[0]:
    # 会話の最初にsystemのcontentを設定
    if 'messages' not in st.session_state:
        st.session_state.messages = [{'role': 'system', 'content': system_content}]
    # 以前のメッセージを表示
    for message in st.session_state.messages:
        if message['role'] != 'system':
            with st.chat_message(message['role']):
                st.markdown(message['content'])
                # メッセージからマークダウンリンクを抽出
                markdown_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', message['content'])
                # 抽出したマークダウンリンクをリンクボタンとして表示
                for title, url in markdown_links:
                    st.link_button(title, url)
    # ユーザーからの新しい入力を取得
    if prompt := st.chat_input('お話ししましょう。'):
        # 直近のユーザーメッセージを取得
        recent_user_messages = [message['content'] for message in st.session_state.messages[-1*st.session_state['history_num']:] if message['role'] == 'user' and message['content'].strip() != '']
        # それらのメッセージを一つのテキストに結合
        combined_prompt = ' '.join(recent_user_messages)
        # 結合したテキストをベクトル化して知識を取得
        if combined_prompt.strip() != '':
            knowledge = mf.get_knowledge(client, combined_prompt, st.session_state['answer_num'], df)
        else:
            knowledge = mf.get_knowledge(client, prompt, st.session_state['answer_num'], df)
        prompt_user = prompt
        prompt_api = prompt
        prompt_api += '\n---\nなお以下は上記に関連しているかもしれない情報です。良ければ回答の参考にしてください。関連している場合は、リンクを回答に含めてくれると嬉しいです。\n' + knowledge
        # ユーザーへの表示用メッセージを更新
        st.session_state.messages.append({'role': 'user', 'content': prompt_user})
        # API用メッセージを更新
        if 'messages_api' not in st.session_state:
            st.session_state.messages_api = [{'role': 'system', 'content': system_content}]
        st.session_state.messages_api.append({'role': 'user', 'content': prompt_api})
        if len(st.session_state.messages_api) > 5:
            st.session_state.messages_api.pop(0)
        with st.chat_message('user'):
            st.markdown(prompt_user)
        full_response = ''
        with st.chat_message('assistant'):
            with st.spinner('AIが応答を生成中です……'):
                message_placeholder = st.empty() # 一時的なプレースホルダーを作成
                try:
                    # ChatGPTからのストリーミング応答を処理
                    for response in client.chat.completions.create(
                        model=st.session_state['model'],
                        messages=[
                            {'role': m['role'], 'content': m['content']}
                            for m in st.session_state.messages_api
                        ],
                        stream=True):
                        full_response += response.choices[0].delta.content or ''
                        message_placeholder.markdown(full_response + '▌') # レスポンスの途中結果を表示
                    message_placeholder.markdown(full_response) # 最終レスポンスを表示
                    # メッセージからマークダウンリンクを抽出
                    markdown_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', full_response)
                    # 抽出したマークダウンリンクをリンクボタンとして表示
                    for title, url in markdown_links:
                        st.link_button(title, url)
                except BadRequestError as e:
                    mf.show_fail_toast('不幸なことに、エラーが起きてしまいました。知識数が多すぎる可能性があります。サイドバー上で少し減らしつつ、「再読み込み」ボタンを押すといいかもしれません…')
        st.session_state.messages.append({'role': 'assistant', 'content': full_response}) # 応答をメッセージに追加
        st.session_state.messages_api.append({'role': 'assistant', 'content': full_response}) # 応答をメッセージに追加
        if len(st.session_state.messages_api) > 5:
            st.session_state.messages_api.pop(0)

# -------------------------
# 学習データ
# -------------------------
elif page == pages[1]:
    tab1, tab2, tab3, tab4 = st.tabs(['知識の変更', 'ダッシュボード', '全データ (簡易)', '全データ (詳細)'])
    review_data_count = df['strong_point_from_review'].count()
    no_user_knowledge_warning = '追加された知識がありません。'
    with tab2:
        st.write('##### 🤔 メトリクス')
        col1, col2, col3 = st.columns(3)
        with col1:
            data_num = len(df)
            if len(user_df) == 0:
                st.metric('合計知識数', data_num)
            else:
                st.metric('合計知識数', data_num, delta=str(len(user_df)) + ' 件')
        with col2:
            st.metric('クチコミ分析数', review_data_count)
        with col3:
            total_items = df['strong_point_from_review'].dropna().apply(lambda x: len(x.split(','))).sum()
            st.metric('クチコミから抽出された魅力の数', total_items)
        st.divider()
        st.write('##### 📊 知識の関連性マップ')
        D = st.selectbox('次元', ['3D', '2D'])
        # name_embeddedとdescription_embeddedを一つのリストにまとめる
        name_embeddings = df['name_embedded'].dropna().tolist()
        description_embeddings = df['description_embedded'].dropna().tolist()
        embeddings = name_embeddings + description_embeddings
        # nameデータをname_embeddedとdescription_embeddedのデータが存在する行に限定
        names = df['name'][df['name_embedded'].notna()].tolist() + df['name'][df['description_embedded'].notna()].tolist()
        # PCAで次元圧縮
        pca = PCA(n_components=3 if D == '3D' else 2)
        pca_result = pca.fit_transform(embeddings)
        # 散布図のデータを作成
        chart_data = pd.DataFrame({
            'x_embedded': pca_result[:, 0],
            'y_embedded': pca_result[:, 1],
            'z_embedded': pca_result[:, 2] if D == '3D' else None,
            'type': ['name_embedded'] * len(name_embeddings) + ['description_embedded'] * len(description_embeddings),
            'name': names,
            'id': df[partition_key_name][df['name_embedded'].notna()].tolist() + df[partition_key_name][df['description_embedded'].notna()].tolist()
        })
        # 3Dまたは2D散布図を作成
        if D == '3D':
            fig = px.scatter_3d(chart_data, x='x_embedded', y='y_embedded', z='z_embedded', color='type')
        else:
            fig = px.scatter(chart_data, x='x_embedded', y='y_embedded', color='type')
        # 散布図の設定を更新
        fig.update_traces(marker=dict(size=2), hovertemplate='id: %{customdata[1]}<br>name: %{customdata[0]}<extra></extra>', customdata=chart_data[['name', 'id']].values.tolist())
        # 散布図を表示
        st.plotly_chart(fig)
        st.caption('''
        - nameとdescriptionの内容を数値化することで、AIの知識を管理しています。この散布図はその数値を圧縮しプロットしたものです。
        - この散布図で孤立しすぎている知識は、あまり有効に利用できていない、またはAIのノイズになっている可能性があります。特に、descriptionのマーカーで孤立しすぎているものは一度確認してみてください。
        - 3Dの方が高精度です。
        ''')
    with tab1:
        # 追加知識の入力
        with st.expander('##### ✍ 追加', expanded=True):
            col1, col2 = st.columns(2)
            # user_dfが空の場合はIDを1から割り振り、空でない場合は既存のIDの最大値の次の数字から順に割り振る
            if user_df.empty:
                max_id = 0
            else:
                max_id = user_df[partition_key_name].max()
            with col1:
                additional_info_name = st.text_input('知識名 (主題)', placeholder='平均気温')
            with col2:
                additional_info_url = st.text_input('URL', placeholder='https://www.hida-kankou.jp/', help='この項目はオプションです。')
            additional_info_desc = st.text_area('詳細', placeholder='飛騨市の気温は1年を通して…', help='ご存じの場合は、マークダウン記法で入力すると品質が向上する可能性があります。ただし、見出しに関しては1と2(#と##)は使わず、3(###)以降を用いてください。')
            mf.init_session('add_button_state', '追加')
            if st.session_state['add_button_state']:
                if additional_info_name.strip() and additional_info_desc.strip():
                    user_knowledge = pd.DataFrame({
                        'name': [additional_info_name],
                        'description': [additional_info_desc],
                        'url': [additional_info_url]
                    })
                    user_knowledge = mf.process_dataframe(user_knowledge, client)
                    # 新しいデータにIDを割り当てる
                    user_knowledge[partition_key_name] = max_id + 1
                    mf.write_to_dynamodb(dynamodb, user_knowledge, user_table_name)
                    mf.all_reload('add_button_state')
                else:
                    st.warning(forget_input)
            st.divider()
            uploaded_file = st.file_uploader('CSVまたはPDFファイルから知識を一括追加できます。', type=['csv', 'pdf'], help='CSV形式のファイルを正しく読み込むためには、name、description、urlの3列を用意する必要があります。')
            if uploaded_file is not None:
                if uploaded_file.type == 'text/csv':
                    df_uploaded = pd.read_csv(uploaded_file)
                    df_uploaded['id'] = range(int(max_id) + 1, int(max_id) + 1 + len(df_uploaded))
                    df_uploaded = mf.process_dataframe(df_uploaded, client)
                    mf.write_to_dynamodb(dynamodb, df_uploaded, user_table_name)
                elif uploaded_file.type == 'application/pdf':
                    df_uploaded = mf.read_from_pdf(uploaded_file, id=max_id+1)
                    df_uploaded = mf.process_dataframe(df_uploaded, client)
                    mf.write_to_dynamodb(dynamodb, df_uploaded, user_table_name)
                else:
                    mf.show_fail_toast()
            st.caption('''
            - この機能は使用できますが、まだ調整中です。
            - アップロードしたファイルが表示され続ける場合は、右のバツをクリックして消してください。その後の知識の追加が正常に行えません。
            - この機能で知識を追加した後は、必ずサイドバーの「再読み込み」ボタンを押してください。
            - 大規模なファイルをアップロードすると処理に時間がかかります。エラーが出ない限りは、気長に待ってみてください。
            - 現在、サンプルとして飛騨市公式観光サイト「[飛騨の旅](https://www.hida-kankou.jp/)」の各種データを追加してあります。データをまとめたCSVファイルはここに置いてあります。これらのファイルから、知識削除後の再追加も可能です。
            ''')
        # 削除する知識の指定
        with st.expander('##### 🗑️ 削除', expanded=True):
            if not user_df.empty:
                # 削除する知識の指定
                ids_to_delete = st.multiselect('削除したい知識のidを選択してください。', options=user_df[partition_key_name].tolist(), placeholder='idを選択', help='全削除する場合は、idの入力は不要です。入力されていても無視されます。')
                col1, col2 = st.columns(2)
                with col1:
                    mf.init_session('delete_button_state', '削除')
                    if st.session_state['delete_button_state']:
                        if not ids_to_delete:
                            st.warning(forget_input)
                        else:
                            for id in ids_to_delete:
                                mf.delete_item(dynamodb, user_table_name, partition_key_name, id)
                            mf.all_reload('reload_button_state')
                with col2:
                    mf.init_session('delete_all_button_state', '全削除')
                    if st.session_state['delete_all_button_state']:
                        mf.delete_all_items(dynamodb, user_table_name, partition_key_name)
                        mf.all_reload('delete_all_button_state')
            else:
                st.warning(no_user_knowledge_warning)
        
        with st.expander('##### 👌 追加した知識の確認', expanded=True):
            # user_dfが空でない場合のみ、追加済の知識を表示
            if not user_df.empty:
                st.dataframe(user_df, hide_index=True, column_order=(partition_key_name, 'name', 'description', 'url'))
            else:
                st.warning(no_user_knowledge_warning)
            st.caption('反映されない場合は、サイドバーの「再読み込み」ボタンを押してください。')

    with tab3:
        with st.expander('##### ✍ 追加された知識のデータ', expanded=True):
            if user_df.empty:
                st.warning(no_user_knowledge_warning)
            else:
                st.dataframe(user_df, hide_index=True, column_order=(partition_key_name, 'name', 'description'))
        with st.expander('##### 📚 事前学習した知識のデータ', expanded=True):
            st.dataframe(df_base, hide_index=True, column_order=(partition_key_name, 'name', 'description'))

    with tab4:
        with st.expander('##### ✍ 追加された知識のデータ', expanded=True):
            if user_df.empty:
                st.warning(no_user_knowledge_warning)
            else:
                st.dataframe(user_df, hide_index=True)
        with st.expander('##### 📚 事前学習した知識のデータ', expanded=True):
            st.dataframe(df_base, hide_index=True)


# -------------------------
# 投稿文の自動生成
# -------------------------
elif page == pages[2]:
    mf.deco()
    generated_text = 'ここに文章が表示されます。'
    # 列の設定
    col1, col2, col3 = st.columns(3)
    with col1:
        # プラットフォーム選択欄
        platform = st.selectbox('プラットフォーム', ['Instagram', 'Facebook', 'ホームページ'])
    with col2:
        # 言語選択欄
        language = st.selectbox('言語', ['日本語', '英語'])
    with col3:
        # サンプリング温度の設定
        temperature = st.slider('サンプリング温度', max_value = 20, value = 4, help='低いほど実務的で冷静な文章を生成しますが、一方で柔軟性が失われやすくなります。高くするとよりクリエイティブになりますが、文章が破綻する可能性が上がります。初期設定は4です。')
    # テーマ入力欄
    command = st.text_input('指示 / テーマ')
    # 生成ボタン
    if st.button('生成'):
        if command.strip():
            with st.spinner('AIが文章を生成中です……'):
                knowledge = mf.get_knowledge(client, command, st.session_state['answer_num'], df)
                # プロンプトに追加情報を追加
                prompt = f'''
                「{command}」を{platform}で紹介する文章をjson形式で一つだけ提案してください。サンプリング温度は{str(temperature)}、言語は{language}でお願いします。最高の品質で返答してくださるとすごく嬉しいです。
                ---
                jsonスキーマは次の通りです。
                {{"sentence": "紹介する文章"}}
                ---
                なお以下は上記に関連しているかもしれない情報です。良ければ回答の参考にしてください。
                {knowledge}
                '''
                # GPT APIを叩く
                response = client.chat.completions.create(
                    model=st.session_state['model'],
                    messages=[
                        {'role': 'system', 'content': system_content},
                        {'role': 'user', 'content': prompt}
                    ],
                    response_format={ "type": "json_object" }
                    )
                generated_text = json.loads(response.choices[0].message.content)["sentence"].split('#', 1)[0]
                generated_text = generated_text + '\n#飛騨観光'
                mf.show_success_toast()
        else:
            st.warning(forget_input)
    # 生成された文章の表示・編集欄
    generated_text = st.text_area('生成された文章', generated_text)
    # プレビュー
    st.write('''
    ---
    <span style='color: gray; '>**プレビュー**</span>
    ''', unsafe_allow_html=True)
    st.code(generated_text, language='text')
