from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
import numpy as np
from surprise import  Reader,Dataset 
from surprise import KNNBasic 
from surprise.model_selection import cross_validate
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors
from sklearn.metrics.pairwise import sigmoid_kernel
import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from collections import defaultdict
from recommendation_system.config.config import Config
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from typing import List, Dict, Union



#تابع يقوم بتنظيف عمود من البيانات
def clean(text):
  nltk.download('punkt')
  nltk.download('stopwords')
  nltk.download('wordnet') 
  stemmer = nltk.SnowballStemmer("english") 
  stopword=set(stopwords.words('english'))
  text = str(text).lower() 
  text = re.sub('\[.*?\]', '', text) 
  text = re.sub('https?://\S+|www\.\S+', '', text) 
  text = re.sub('<.*?>+', '', text) 
  text = re.sub('[%s]' % re.escape(string.punctuation), '', text) 
  text = re.sub('\n', '', text) 
  text = re.sub('\w*\d\w*', '', text) 
  text = [word for word in text.split(' ') if word not in stopword] 
  text=" ".join(text) 
  text = [stemmer.stem(word) for word in text.split(' ')] 
  text=" ".join(text) 
  return text


#تابع توصية معتمد على المحتوى يعطي توصيات بناء على اسم عمود والخاصية ا
#يأخذ اسم العمود والخاصية كمدخل ويعطي اقتراحات بناء على الخواص الأخرى 
# # Load the dataset 
# metadata = pd.read_csv('data/products.csv', low_memory=False)
# recommendation = content_based_simple('category','sports', metadata, ['category', 'brand'])
# print(recommendation)
def content_based_simple(column, attribute, metadata, features):
#   attribute = clean(attribute)
  content= metadata[features]
  content= content.dropna()
  content = content.assign(combined='')
  for attr in features:
    # content.loc[:, attr] = content[attr].apply(clean)
    if  pd.api.types.is_string_dtype(content[attr]):
      content.loc[:, 'combined'] = content['combined'] + ' ' + content[attr]
  
  tfidf = TfidfVectorizer(stop_words='english')
  tfidf_matrix = tfidf.fit_transform(content['combined'])
  
  cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
  # Get the index of the pro that matches the attribute 
  indices = pd.Series(content.index, index=content[column]).drop_duplicates()
  idx = indices[attribute]
  # Get the pairwise similarity scores of all products with that product 
  sim_scores = list(enumerate(cosine_sim[idx]))
  # Get the scores of the 10 most similar products 
  sim_scores = sim_scores[1:11]
  # Get the product indices 
  product_indices = [i[0] for i in sim_scores]
   # Return the top 10 most similar products 
  return content[column].iloc[product_indices]

 
#تابع يقوم باعادة المنتجات الأكثر شعبية
#يعتمد التابع على تقييمات المستخدمين للمنتج

def mostPopular(df,user_id):
  df = df.drop_duplicates(subset=['user_id', 'product_id'])  #معالجة تكرار التقيم 
  counts = df['user_id'].value_counts()
  data = df[df['user_id'].isin(counts[counts >= 0].index)]
  data.groupby('product_id')['rating'].mean().sort_values(ascending=False)
  # إنشاء مصفوفة تقييمات المستخدمين-المنتجات
  # تحويل البيانات إلى شكل مصفوفة (المستخدمين × المنتجات)
  final_ratings = data.pivot(index = 'user_id', columns ='product_id', values = 'rating').fillna(0)
  # حساب كثافة المصفوفة
  num_of_ratings = np.count_nonzero(final_ratings)
  possible_ratings = final_ratings.shape[0] * final_ratings.shape[1]
  density = (num_of_ratings/possible_ratings) * 100
  # تحضير بيانات التدريب
  final_ratings_T = final_ratings.transpose()
  grouped = data.groupby('product_id').agg({'user_id': 'count'}).reset_index()
  grouped.rename(columns={'user_id': 'score'}, inplace=True)
  training_data = grouped.sort_values(['score', 'product_id'], ascending=[0,1])
  training_data['Rank'] = training_data['score'].rank(ascending=0, method='first')
  recommendations = training_data.head(20)
  #تابع التوصية
  def recommend(id):
    recommend_products = recommendations
    recommend_products.loc[:, 'user_id'] = id 
    column = recommend_products.columns.tolist()
    column = column[-1:] + column[:-1]
    recommend_products = recommend_products[column]
    return recommend_products
  return recommend(user_id)

#تابع يقوم بالتوصية بمنتج معين أو مستخدم مشابه أو الاثنين معا بناء على المميزات
#يجب ان تكون المميزات على شكل onehotencoding H, scaling للحصول على نتائج
# يتم استخدام مقياس min max لتقليل القيم 
# يتم استخدام KNN للتوصية
# المميزات features على شكل dataframe
# يمكن تعديل التابع بحيث يتم حفظ النموذج وإعادة استخدامه

def get_recomend_with_features_knn(df, features, column_name, item_name):
    # التحقق من وجود item_name في DataFrame
    if item_name not in df[column_name].values:
        # raise ValueError(f"'{item_name}' غير موجود في العمود '{column_name}'")
        return []  
    min_max_scaler = MinMaxScaler()
    features = min_max_scaler.fit_transform(features)
    model = neighbors.NearestNeighbors(n_neighbors=6, algorithm='ball_tree')
    model.fit(features)
    dist, idlist = model.kneighbors(features)  
    # الحصول على item_id مع التحقق من وجوده
    item_id = df[df[column_name] == item_name].index
    if len(item_id) == 0:
        # raise ValueError(f"لا يوجد عنصر مطابق لـ '{item_name}'")
        return [] 
    item_id = item_id[0]
    
    # التحقق من أن item_id ضمن النطاق الصحيح
    if item_id >= len(idlist):
        # raise IndexError(f"معرف العنصر {item_id} خارج نطاق بيانات الجيران")
        return []  
    item_list_name = []
    for newid in idlist[item_id]:
      try:
        # التحقق من أن newid صالح قبل الوصول إليه
        if newid < len(df):
            item_list_name.append(df.loc[newid])
        else:
            # print(f"تحذير: معرف {newid} خارج نطاق DataFrame")
            return []
      except:
        continue
    return item_list_name


#تابع يقوم بالتوصية بناء على المحتوى باستخدام cosine_similitry
def get_recomend_with_cosine(df, features,column_name, item_name,num_recommendations=10):
  try:
    scaler = MinMaxScaler()
    pro_features= scaler.fit_transform(features)
    if item_name not in df[column_name].values: 
      # print(f"'{item_name}' not found in the dataset. Please enter a valid item name.") 
      return pd.DataFrame(columns=['product_id'])
    input_item_index = df[df[column_name] == item_name].index[0]
    similarity_scores = cosine_similarity([pro_features[input_item_index]],pro_features )
    similar_item_indices = similarity_scores.argsort()[0][::-1][1:num_recommendations + 1]
    content_based_recommendations = df.iloc[similar_item_indices]
    return content_based_recommendations
  except:
    return pd.DataFrame(columns=['product_id'])


# تابع يقوم بالتوصية بناء على المحتوى لمنتج معين بناء على وصف أو مراجعة نصية لهذا المنتج
def get_recomend_with_describition(df,column_product ,product, review_text):
    tfv = TfidfVectorizer(
        min_df=3, 
        max_features=None, 
        strip_accents='unicode', 
        analyzer='word',
        token_pattern=r'\w{1,}', 
        ngram_range=(1, 3), 
        stop_words='english'
    )
    tfv_matrix = tfv.fit_transform(df[review_text])
    
    # حساب مصفوفة التشابه باستخدام sigmoid_kernel
    sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
    
    # الحصول على الفهرس (index) للمنتج المطلوب
    indices = pd.Series(df.index, index=df[column_product]).drop_duplicates()
    idx = indices[product]
    
    # الحصول على قائمة التشابه للمنتج مع باقي المنتجات
    sig_scores = list(enumerate(sig[idx]))  # [(index, similarity_array), ...]
    
    # تصحيح الخطأ: استخراج قيمة واحدة من المصفوفة (مثل np.max أو np.mean)
    sig_scores = sorted(sig_scores, key=lambda x: np.mean(x[1]), reverse=True)  # نستخدم np.max هنا
    
    # اختيار أفضل 10 توصيات (تجاهل التشابه مع نفسه إذا كان في النتائج)
    sig_scores = sig_scores[1:11]
    
    # استخراج indices المنتجات الموصى بها
    pro_indices = [i[0] for i in sig_scores]
    
    # إرجاع أسماء المنتجات الموصى بها
    return df[column_product].iloc[pro_indices]

#تابع يقوم بحساب وزن للمنتجات حسب تاريخ الإصدار
#الوزن بتناقص مع زيادة الفترة الزمنية بين تاريخ الإصدار واليوم
def calculate_weighted_popularity(release_date):
    # Convert the release date to datetime object 
    release_date = datetime.strptime(release_date, '%Y-%m-%d') 
    # Calculate the time span between release date and today's date 
    time_span = datetime.now() - release_date 
    # Calculate the weighted popularity score based on time span (e.g., more recent releases have higher weight) 
    weight = 1 / (time_span.days + 1) 
    return weight

#  دالة لترتيب المنتجات وإرجاع أحدث n منتج
def get_top_n_products_by_weight(df, date_column, n):
    # تطبيق الدالة على عمود التاريخ وإضافة العمود الجديد
    df['weighted_score'] = df[date_column].apply(calculate_weighted_popularity)
    # ترتيب المنتجات تنازليًا حسب الوزن
    sorted_df = df.sort_values(by='weighted_score', ascending=False)
    
    # اختيار أفضل n منتج
    top_n_products = sorted_df.head(n)
    
    return top_n_products

 
# دالة تجميع لهياكل البيانات حسب قيم رقمية لأعمدة معينة مثل العمر السعر 
def cluster_data(
    df: pd.DataFrame,
    features: list,
    n_clusters: int = 10,
    auto_optimize_k: bool = False,
    max_k: int = 15,
    random_state: int = 42
) -> pd.DataFrame:
    """
    دالة لتجميع البيانات باستخدام K-Means مع تحسين عدد المجموعات تلقائيًا (اختياري).
    
    Parameters:
        df (pd.DataFrame): هيكل البيانات المدخل.
        features (list): قائمة بالأعمدة المستخدمة في التجميع (مثل ['age'] أو ['price', 'production_date']).
        n_clusters (int): عدد المجموعات المطلوبة (إفتراضيًا 10).
        auto_optimize_k (bool): إذا كان True، يتم البحث عن العدد الأمثل للمجموعات (إفتراضيًا False).
        max_k (int): الحد الأقصى لعدد المجموعات عند التحسين التلقائي (إفتراضيًا 15).
        random_state (int): للتكرارية (إفتراضيًا 42).
    
    Returns:
        pd.DataFrame: هيكل البيانات مع عمود جديد 'cluster' يوضح المجموعة لكل عنصر.
    """
    
    # نسخ البيانات لتجنب التعديل على الأصل
    df = df.copy()
    
    # 1. التحقق من وجود الأعمدة المطلوبة
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        raise ValueError(f"الأعمدة التالية غير موجودة: {missing_cols}")
    
    # 2. تطبيع البيانات
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df[features])
    
    # 3. التحسين التلقائي لعدد المجموعات (إن طُلب)
    if auto_optimize_k:
        best_k = 2
        best_score = -1
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=random_state)
            labels = kmeans.fit_predict(normalized_data)
            score = silhouette_score(normalized_data, labels)
            if score > best_score:
                best_score = score
                best_k = k
        n_clusters = best_k
        # print(f"تم العثور على العدد الأمثل للمجموعات: {best_k} (Score: {best_score:.2f})")
    
    # 4. التجميع باستخدام K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    df['cluster'] = kmeans.fit_predict(normalized_data)
    
    # 5. إضافة معلومات إضافية عن المجموعات
    cluster_stats = df.groupby('cluster')[features].agg(['mean', 'count'])
    # print("\nإحصائيات المجموعات:")
    # print(cluster_stats)
    
    return df

# دالة تقوم بإرجاع توصيات بناء على وصف يدخله المستخدم ونوع المنتج

def recommend_products_with_reviewFromUser(reviews_df, products_df, user_review, user_category, top_n=5):
    """
    يقترح منتجات بناءً على مراجعة المستخدم والفئة المحددة
    
    Args:
        reviews_df (DataFrame): يحتوي على user_id, product_id, rating, review_text
        products_df (DataFrame): يحتوي على product_id, category, price, brand
        user_review (str): نص المراجعة من المستخدم
        user_category (str): الفئة المطلوبة
        top_n (int): عدد التوصيات المطلوبة
        
    Returns:
        DataFrame: المنتجات الموصى بها مع معلوماتها
    """
    
    # 1. دمج البيانات
    merged_df = pd.merge(reviews_df, products_df, on='product_id')
    
    # 2. تصفية حسب الفئة المطلوبة
    category_products = merged_df[merged_df['category'].str.lower() == user_category.lower()]
    
    if len(category_products) == 0:
        return pd.DataFrame()  # إرجاع DataFrame فارغ إذا لم توجد منتجات في الفئة
    
    # 3. معالجة النصوص
    tfidf = TfidfVectorizer(stop_words='english')
    review_matrix = tfidf.fit_transform(category_products['review_text'])
    user_review_vec = tfidf.transform([user_review])
    
    # 4. حساب التشابه
    similarity_scores = cosine_similarity(user_review_vec, review_matrix).flatten()
    category_products['similarity'] = similarity_scores
    
    # 5. حساب درجة مركبة (التشابه + التقييم)
    scaler = MinMaxScaler()
    category_products['norm_rating'] = scaler.fit_transform(category_products[['rating']])
    category_products['composite_score'] = 0.7 * category_products['similarity'] + 0.3 * category_products['norm_rating']
    
    # 6. ترتيب النتائج
    recommendations = category_products.sort_values('composite_score', ascending=False)
    
    # 7. إزالة التكرارات وإرجاع أفضل النتائج
    final_recommendations = recommendations.drop_duplicates('product_id').head(top_n)
    
    return final_recommendations[['product_id', 'category', 'brand', 'price', 'rating', 'similarity']]

  
#دلة تقوم باقتراح توصيات بناء على وصف يدخله المستخدم وفئة (اختياري) 
def preprocess_text(text):
    """
    معالجة النص لإزالة التشويش وتحسين جودة المقارنة
    """
    if not isinstance(text, str) or pd.isna(text):
        return ""
    # التحويل لحروف صغيرة
    text = text.lower()
    
    # إزالة علامات الترقيم والأرقام
    text = re.sub(f'[{string.punctuation}0-9]', ' ', text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # إزالة stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

def recommend_products_by_description(products_df, user_description,description_column, top_n=5, category_filter=None):
    """
    يقترح منتجات بناءً على وصف المستخدم
    
    Args:
        products_df (DataFrame): يحتوي على product_id, product_name, category, description
        user_description (str): وصف المنتج المطلوب من المستخدم
        top_n (int): عدد التوصيات المطلوبة
        category_filter (str): فلتر اختياري حسب الفئة
        
    Returns:
        DataFrame: المنتجات الموصى بها مع درجة التشابه
    """
    nltk.download('punkt_tab')
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    # 1. نسخ البيانات لتجنب التعديل على الأصل
    df = products_df.copy()
    
    # 2. تطبيق فلتر الفئة إذا تم تحديده
    if category_filter:
        df = df[df['category'].str.lower() == category_filter.lower()]
    
    if len(df) == 0:
        return pd.DataFrame
    
    # 3. معالجة النصوص
    df['processed_desc'] = df[description_column].apply(preprocess_text)
    processed_user_desc = preprocess_text(user_description)
    
    # 4. إنشاء نموذج TF-IDF
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    tfidf_matrix = tfidf.fit_transform(df['processed_desc'])
    user_desc_matrix = tfidf.transform([processed_user_desc])
    
    # 5. حساب التشابه
    cosine_sim = cosine_similarity(user_desc_matrix, tfidf_matrix)
    df['similarity_score'] = cosine_sim[0]
    
    # 6. ترتيب النتائج وإرجاع الأفضل
    recommendations = df.sort_values('similarity_score', ascending=False).head(top_n)
    
    return recommendations

# إنشاء تابع لاستنتاج اهتمامات المستخدم من البيانات
def extract_user_interests(users_df, reviews_df, interactions_df, products_df):
    """
    تابع لاستنتاج اهتمامات المستخدمين بناءً على مراجعاتهم وتفاعلاتهم مع المنتجات
    
    المعاملات:
        users_df: DataFrame يحتوي على بيانات المستخدمين (user_id, age, gender, location)
        reviews_df: DataFrame يحتوي على المراجعات (user_id, product_id, rating, review_text)
        interactions_df: DataFrame يحتوي على تفاعلات المستخدمين (user_id, product_id, interaction_type, timestamp)
        products_df: DataFrame يحتوي على بيانات المنتجات (product_id, category, price, brand)
        
    المخرجات:
        DataFrame معدل لبيانات المستخدمين مع إضافة أعمدة الاهتمامات
    """
    
    # دمج البيانات للحصول على معلومات كاملة
    reviews_with_products = pd.merge(reviews_df, products_df, on='product_id')
    interactions_with_products = pd.merge(interactions_df, products_df, on='product_id')
    
    # إنشاء قاموس لتخزين اهتمامات كل مستخدم
    user_interests = defaultdict(lambda: {
        'categories': defaultdict(int),
        'price_range': [],
        'brands': defaultdict(int),
        'interaction_types': defaultdict(int)
    })
    
    # تحليل المراجعات لاستنتاج الاهتمامات
    for _, row in reviews_with_products.iterrows():
        user_id = row['user_id']
        user_interests[user_id]['categories'][row['category']] += row['rating']
        user_interests[user_id]['brands'][row['brand']] += row['rating']
        user_interests[user_id]['price_range'].append(row['price'])
    
    # تحليل التفاعلات لاستنتاج الاهتمامات
    for _, row in interactions_with_products.iterrows():
        user_id = row['user_id']
        user_interests[user_id]['categories'][row['category']] += 1
        user_interests[user_id]['brands'][row['brand']] += 1
        user_interests[user_id]['interaction_types'][row['interaction_type']] += 1
        user_interests[user_id]['price_range'].append(row['price'])
    
    # استخلاص الاهتمامات الرئيسية لكل مستخدم
    def get_top_interests(interests_dict, n=3):
        if not interests_dict:
            return []
        sorted_items = sorted(interests_dict.items(), key=lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_items[:n]]
    # إعداد البيانات للإضافة إلى DataFrame المستخدمين
    users_data = []
    for user_record in users_df.to_dict('records'):
        user_id = user_record['user_id']
        interests = user_interests.get(user_id, {})
        
        # الفئات المفضلة
        top_categories = get_top_interests(interests.get('categories', {}))
        
        # نطاق السعر
        price_range = interests.get('price_range', [])
        avg_price = round(sum(price_range)/len(price_range), 2) if price_range else 0
        
        # الماركات المفضلة
        top_brands = get_top_interests(interests.get('brands', {}))
        
        # أنواع التفاعلات
        top_interactions = get_top_interests(interests.get('interaction_types', {}))
        
        # تحديث سجل المستخدم
        updated_record = user_record.copy()
        updated_record.update({
            'interests_categories': top_categories,
            'avg_preferred_price': avg_price,
            'preferred_brands': top_brands,
            'common_interaction_types': top_interactions
        })
        users_data.append(updated_record)
    # إنشاء DataFrame جديد مع بيانات الاهتمامات
    updated_users_df = pd.DataFrame(users_data)
    return updated_users_df

  
# تابع يقوم بإيجاد المستخدمين المشابهين لمستخدم معين بناء على صفاته
def get_similar_user_from_userfeatures(user_id):
    df = pd.read_csv(Config.DATA_PATHS['users'])
    user_scaler = StandardScaler()
    df[['age']] = user_scaler.fit_transform(df[['age']])
    user_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    user_categorical = user_encoder.fit_transform(df[['gender', 'location']])

    pro_num = pd.DataFrame(df[['age']].values)
    pro_cat = pd.DataFrame(user_categorical)
   

    if user_id not in df['user_id'].values:
        # raise ValueError(f"user_id غير موجود في بيانات المستخدمين")
        return []
    features = pd.concat([pro_num,pro_cat], axis=1)
    df = df.reset_index(drop=True)
    
    min_max_scaler = MinMaxScaler()
    features = min_max_scaler.fit_transform(features)
    model = neighbors.NearestNeighbors(n_neighbors=6, algorithm='ball_tree')
    model.fit(features)
    dist, idlist = model.kneighbors(features)
    
    # الحصول على item_id مع التحقق من وجوده
    item_id = df[df['user_id'] == user_id].index
    if len(item_id) == 0:
        # raise ValueError(f"لا يوجد عنصر مطابق لـ '{item_name}'")
        return []
    
    item_id = item_id[0]
    
    # التحقق من أن item_id ضمن النطاق الصحيح
    if item_id >= len(idlist):
        # raise IndexError(f"معرف العنصر {item_id} خارج نطاق بيانات الجيران")
        return []
    
    item_list_name = []
    for newid in idlist[item_id]:
        # التحقق من أن newid صالح قبل الوصول إليه
        if newid < len(df):
            item_list_name.append(df.loc[newid])
        else:
            # print(f"تحذير: معرف {newid} خارج نطاق DataFrame")
            continue 
    predication_knn =  [item['user_id'] if isinstance(item, dict) else item.user_id for item in item_list_name] 
    
    return predication_knn


def find_similar_users(target_user_id, users_with_interests_df, reviews_df, interactions_df, n_recommendations=5):
    """
    تابع  لإيجاد المستخدمين الأكثر تشابهًا مع مستخدم معين
    """
    
    # التأكد من وجود المستخدم المستهدف في البيانات
    if target_user_id not in users_with_interests_df['user_id'].values:
        raise ValueError(f"User ID {target_user_id} not found in the dataset")
    
    # استخراج بيانات المستخدم المستهدف
    target_user = users_with_interests_df[users_with_interests_df['user_id'] == target_user_id].iloc[0]
    
    # إنشاء ميزات للتشابه - وظيفة مساعدة
    def prepare_features(user_row):
        user = user_row[1]  # الحصول على بيانات المستخدم من السلسلة
        user_id = user['user_id']
        
        # الميزات العددية مع معالجة القيم الفارغة
        features = {
            'user_id': user_id,
            'age': user.get('age', 0),  # استخدام get مع قيمة افتراضية
            'avg_preferred_price': user.get('avg_preferred_price', 0),
            'n_reviews': reviews_df[reviews_df['user_id'] == user_id].shape[0] if not reviews_df.empty else 0,
            'n_interactions': interactions_df[interactions_df['user_id'] == user_id].shape[0] if not interactions_df.empty else 0,
        }
        
        # ميزات الاهتمامات (تشفير one-hot للمجموعات)
        categories = ['interests_categories', 'preferred_brands', 'common_interaction_types']
        for cat in categories:
            for item in user.get(cat, []):
                # تنظيف أسماء الأعمدة لإزالة أي أحخاص غير مرغوب فيها
                clean_item = str(item).replace(' ', '_').replace('/', '_').replace('\\', '_')
                features[f"{cat}_{clean_item}"] = 1
        
        return features
    
    # إعداد ميزات للمستخدم المستهدف
    target_features = prepare_features(('target', target_user))
    
    # إعداد ميزات لجميع المستخدمين الآخرين
    other_users = users_with_interests_df[users_with_interests_df['user_id'] != target_user_id]
    
    if len(other_users) == 0:
        return pd.DataFrame(columns=['user_id', 'similarity_score'])
    
    # تحضير الميزات للمستخدمين الآخرين
    all_features_list = []
    for _, row in other_users.iterrows():
        all_features_list.append(prepare_features((None, row)))
    
    # إنشاء DataFrames مع معالجة القيم الفارغة
    target_df = pd.DataFrame([target_features]).fillna(0)
    features_df = pd.DataFrame(all_features_list).fillna(0)
    
    # إزالة عمود user_id قبل التحجيم
    user_ids = features_df['user_id'].copy()
    features_df = features_df.drop('user_id', axis=1)
    target_df = target_df.drop('user_id', axis=1)
    
    # محاذاة الأعمدة بين DataFrames باستخدام reindex (بدون حلقات)
    all_columns = target_df.columns.union(features_df.columns)
    target_df = target_df.reindex(columns=all_columns, fill_value=0)
    features_df = features_df.reindex(columns=all_columns, fill_value=0)
    
    # التحقق من وجود قيم NaN بعد المحاذاة
    if target_df.isnull().any().any() or features_df.isnull().any().any():
        # إذا وجدت قيم NaN، نملؤها بصفر
        target_df = target_df.fillna(0)
        features_df = features_df.fillna(0)
    
    # تطبيع البيانات
    scaler = MinMaxScaler()
    combined_data = pd.concat([target_df, features_df], ignore_index=True)
    
    # التحقق من وجود قيم غير عددية
    for col in combined_data.columns:
        if combined_data[col].dtype == 'object':
            # تحويل القيم النصية إلى قيم رقمية إذا وجدت
            combined_data[col] = pd.to_numeric(combined_data[col], errors='coerce').fillna(0)
    
    scaled_data = scaler.fit_transform(combined_data)
    
    target_scaled = scaled_data[0:1]  # بيانات المستخدم المستهدف
    others_scaled = scaled_data[1:]   # بيانات المستخدمين الآخرين
    
    # حساب تشابه جيب التمام
    similarities = cosine_similarity(target_scaled, others_scaled)
    
    # إنشاء DataFrame النتائج
    results = pd.DataFrame({
        'user_id': user_ids,
        'similarity_score': similarities[0]
    })
    
    # إضافة معلومات إضافية عن المستخدمين المشابهين
    results = results.merge(users_with_interests_df, on='user_id')
    
    # ترتيب حسب درجة التشابه
    results = results.sort_values('similarity_score', ascending=False).head(n_recommendations)
    
    return results



#التحقق من تطابق البيانات بين المستخدمين وبين جدول التفاعلات
def check_data_consistency(data):
    users_in_interactions = set(data['interactions']['user_id'].unique())
    users_in_features = set(data['users']['user_id'].unique())
    
    missing = users_in_interactions - users_in_features
    if missing:
        print(f"تحذير: {len(missing)} مستخدمين في التفاعلات ولكن ليس في بيانات المستخدمين")
    
    return not bool(missing)


# تابع لتحميل البيانات ومعالجتها 
def load_and_preprocess_data():
            # Load data
            users = pd.read_csv(Config.DATA_PATHS['users'])
            products = pd.read_csv(Config.DATA_PATHS['products'])
            interactions = pd.read_csv(Config.DATA_PATHS['reviews']).dropna()

            # Ensure we only keep users and products that exist in interactions
            interactions = interactions[
                interactions['user_id'].isin(users['user_id']) &
                interactions['product_id'].isin(products['product_id'])
            ]

            # Get unique user and product IDs that have interactions
            active_users = interactions['user_id'].unique()
            active_products = interactions['product_id'].unique()

            # Filter datasets
            # users = users[users['user_id'].isin(active_users)]
            # products = products[products['product_id'].isin(active_products)]
            sensitive_features = users[['user_id', 'gender']].copy()

            # Process numerical features
            user_scaler = StandardScaler()
            users[['age']] = user_scaler.fit_transform(users[['age']])

            product_scaler = StandardScaler()
            products['price'] = product_scaler.fit_transform(products[['price']])

            # Process categorical features
            user_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            user_categorical = user_encoder.fit_transform(users[['gender', 'location']])

            product_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            product_categorical = product_encoder.fit_transform(products[['category', 'brand','country_made']])

            # Create mapping dictionaries for user and product indices
            user_id_to_idx = {user_id: idx for idx, user_id in enumerate(users['user_id'])}
            product_id_to_idx = {product_id: idx for idx, product_id in enumerate(products['product_id'])}

            # Create interaction matrix
            interaction_matrix = np.zeros((len(users), len(products)))
            for _, row in interactions.iterrows():
                user_idx = user_id_to_idx[row['user_id']]
                product_idx = product_id_to_idx[row['product_id']]
                interaction_matrix[user_idx, product_idx] = row['rating']

            return {
                'users': users,
                'products': products,
                'interactions': interactions,
                'interaction_matrix': interaction_matrix,
                'sensitive_features': sensitive_features,
                'features': {
                    'user_numerical': users[['age']].values,
                    'user_categorical': user_categorical,
                    'product_numerical': products[['price']].values,
                    'product_categorical': product_categorical,
                },
                'encoders': {
                    'user_encoder': user_encoder,
                    'product_encoder': product_encoder,
                    'user_scaler': user_scaler,
                    'product_scaler': product_scaler
                },
                'mappings': {
                    'user_id_to_idx': user_id_to_idx,
                    'product_id_to_idx': product_id_to_idx
                }
            }
 
# تابع يقوم بإرجاع المنتجات التي قام المستخدم بتقيمها والتي هي أعلى من تقييم معين 
def get_high_rated_products(df, user_id, r=3):
    """
    تقوم هذه الدالة بإرجاع قائمة product_id للمنتجات التي قام مستخدم معين بتقييمها بتقييم معين أو أكثر
    
    المعطيات:
        df (DataFrame): بيانات التقييمات تحتوي على الأعمدة ['user_id', 'product_id', 'rating']
        user_id (int): معرّف المستخدم المطلوب
    
    المخرجات:
        list: قائمة بمعرّفات المنتجات التي حصلت على تقييم r+ من هذا المستخدم
    """
    # تصفية البيانات للمستخدم المطلوب والتقييمات 3+
    filtered = df[(df['user_id'] == user_id) & (df['rating'] >= r)]
    
    # إرجاع قائمة بمعرّفات المنتجات
    return filtered['product_id'].tolist()

# تابع يقوم بإرجاع المنتجات التي قام المستخدم بالتفاعل معها
def get_user_interacted_products(interactions_df, user_id):
    """
    تقوم هذه الدالة بإرجاع قائمة بالمنتجات التي تفاعل معها مستخدم معين
    
    المعطيات:
        interactions_df (DataFrame): بيانات التفاعلات تحتوي على الأعمدة ['user_id', 'product_id', 'interaction_type', 'timestamp']
        user_id (int): معرّف المستخدم المطلوب
    
    المخرجات:
        list: قائمة بمعرّفات المنتجات التي تفاعل معها المستخدم
    """
    # تصفية التفاعلات الخاصة بالمستخدم المطلوب
    user_interactions = interactions_df[interactions_df['user_id'] == user_id]
    
    # إزالة التكرارات للحصول على قائمة فريدة من المنتجات
    unique_products = user_interactions['product_id'].unique()
    
    return unique_products.tolist()

# تابع يقوم بإرجاع المنتجات التي قام المستخدم بالتفاعل معها أو تقيمها تقييم مرتفع
def get_user_prefered_product(reviews, interaction, user_id):
    return get_high_rated_products(reviews, user_id) + get_user_interacted_products(interaction, user_id)


#إرجاع DataFrame كامل مع جميع التفاعلات:
def get_user_interactions_details(interactions_df, user_id):
    return interactions_df[interactions_df['user_id'] == user_id]

# إرجاع عدد التفاعلات لكل منتج:
def get_user_interactions_count(interactions_df, user_id):
    user_interactions = interactions_df[interactions_df['user_id'] == user_id]
    return user_interactions['product_id'].value_counts().to_dict()
# تصفية حسب نوع التفاعل
def get_user_interactions_by_type(interactions_df, user_id, interaction_type):
    filtered = interactions_df[(interactions_df['user_id'] == user_id) & 
                             (interactions_df['interaction_type'] == interaction_type)]
    return filtered['product_id'].unique().tolist()


#تابع يقوم بإرجاع معرفات المنتجات التي اهتم بها المستخدمين المشابهين بالاهتمامات للمستخدم

def get_user_similar_prefered_products(user_id, df_user, df_pro,df_inter, df_review ):
    similar_user_products = []
    #فحص لمعرفة إذا كان المستخدم قد تفاعل سابقاً مع الموقع
    if (user_id in df_review['user_id'].values) and (user_id in df_inter['user_id'].values) :
        users_with_interests = extract_user_interests(df_user, df_review, df_inter, df_pro)
        if user_id in users_with_interests['user_id'].values:
            similar_users = find_similar_users(
                target_user_id=user_id,
                users_with_interests_df=users_with_interests,
                reviews_df=df_review,
                interactions_df=df_inter,
                n_recommendations=5
            )
            for row in similar_users.itertuples():
                similar_user_products = similar_user_products + get_user_prefered_product(df_review, df_inter, row.user_id)

        similar_user_products = list(set(similar_user_products))
        return similar_user_products
    else :
        # cold start
        #إرجاع المستخدمين المشابهين للمستخدم بالصفات
        similar_users = get_similar_user_from_userfeatures(user_id)
        print(similar_users)
        for user in similar_users:
          similar_user_products = similar_user_products + get_user_prefered_product(df_review, df_inter, user)
        return similar_user_products


# تابع يرجع المنتجات المقترحة بناء على لمنتجات معينة بناء على التوابع السابقة
from recommendation_system.imageRecommender.imageRecommender import ImageRecommendationService
def get_products_from_product(product_ids):
    df_pro = pd.read_csv(Config.DATA_PATHS['products'], low_memory=False)
    pro_images = pd.read_csv(Config.DATA_PATHS['images_df'], low_memory=False)
    metadata = load_and_preprocess_data()
    df = metadata['products']
    pro_num =pd.DataFrame(metadata['features']['product_numerical'])
    pro_cat =pd.DataFrame(metadata['features']['product_categorical'])
    features = pd.concat([pro_num,pro_cat], axis=1)
    service = ImageRecommendationService()
    image_recommendations = []
    predication_cosine = []
    predication_knn = []
    recommendation_description = []
    filter_brand_category_price = []
    for pro_id in product_ids:
            if pro_id in pro_images['product_id'].values:   
                recommendations = service.get_recommendations(pro_images.loc[pro_images['product_id'] == pro_id,'images'].values[0], 3)
                image_recommendations = image_recommendations + [item['product_id'] for item in recommendations]
            filtered_category = []
            filtered_brand = []
            filter_price= []
            if 'category' in df_pro.columns:
                filtered_cat = df_pro[df_pro['category'] == df_pro.loc[df_pro['product_id'] == pro_id,'category'].values[0]]
                filtered_category = filtered_category + filtered_cat['product_id'].tolist()
            if 'brand' in df_pro.columns:
                filtered_bra = df_pro[df_pro['brand'] == df_pro.loc[df_pro['product_id'] == pro_id,'brand'].values[0]]  
                filtered_brand = filtered_brand +  filtered_bra['product_id'].tolist() 
            if 'price' in df_pro.columns:
                clustered_product_price = cluster_data(df_pro, features=['price'], n_clusters=5)
                filter_pri = clustered_product_price[clustered_product_price['cluster'] == clustered_product_price.loc[clustered_product_price['product_id'] == pro_id,'cluster'].values[0]]
                filter_price = filter_price + filter_pri['product_id'].tolist()
            brand_category_price = list(set(filtered_category) & set(filtered_brand) &  set(filter_price))
            filter_brand_category_price = filter_brand_category_price + brand_category_price[:2]
            pre_knn = get_recomend_with_features_knn(df, features,'product_id',pro_id)  # list
            predication_knn = predication_knn + [item['product_id'] if isinstance(item, dict) else item.product_id for item in pre_knn][:2]


            pre_cosine = get_recomend_with_cosine(df, features,'product_id',pro_id, 2) #dataframe
            predication_cosine = predication_cosine + pre_cosine['product_id'].tolist()
                
            if 'product_name' in df_pro.columns:
                rec_description= recommend_products_by_description(df_pro, user_description=df_pro.loc[df_pro['product_id'] == pro_id,'product_name'].values[0],description_column="product_name",top_n=2 )
                recommendation_description= recommendation_description + rec_description['product_id'].tolist()
    image_recommendations= list(set(image_recommendations))
    filter_brand_category_price= list(set(filter_brand_category_price))
    predication_knn = list(set(predication_knn))
    predication_cosine = list(set(predication_cosine))
    recommendation_description = list(set(recommendation_description))
    recomation_products ={
        'image_recommendations': image_recommendations,
        'same_category_brand_avgprice': filter_brand_category_price,
        'knn': predication_knn,
        'cosine': predication_cosine,
        'description_based': recommendation_description
    }

    return recomation_products



## تابع يعطي اقتراحات لمستخدم معين

def suggest_for_user(user_id):
    df_user = pd.read_csv(Config.DATA_PATHS['users'], low_memory=False)
    df_pro = pd.read_csv(Config.DATA_PATHS['products'], low_memory=False)
    df_inter = pd.read_csv(Config.DATA_PATHS['interactions'], low_memory=False)
    df_review = pd.read_csv(Config.DATA_PATHS['reviews'], low_memory=False)
    #فحص لمعرفة إذا كان المستخدم قد تفاعل سابقاً مع الموقع
    if  (user_id in df_inter['user_id'].values) :
        user_product = []
        user_product = get_user_interacted_products(df_inter, user_id)
        if (user_id in df_review['user_id'].values):
              user_product = user_product +  get_high_rated_products(df_review, user_id)
        # user_products_based= get_products_from_product(user_product)
        user_products_based= get_cached_products_recommendations(user_product)
        if not user_products_based:
            user_products_based= get_products_from_product(user_product)
    else:
        most_popular = mostPopular(df_review,11)
        list_most_popular =  most_popular['product_id'].tolist()
        user_products_based ={'most_popular': list_most_popular} 
    similar_users_based= get_user_similar_prefered_products(user_id, df_user, df_pro,df_inter, df_review )
    recommendation = {
      "user_products_based": user_products_based,
       "similar_users_based": similar_users_based
    }
    return recommendation





##  تابع يعطي اقتراحات لمستخدم معين مع تحليل عميق
def deep_suggest_for_user(user_id):
    # إيجاد المنتجات التي اهتم بها المستخدم أو المستخدمين الذين لديهم نفس الاهتمامات
    sugeset_product = suggest_for_user(user_id)
    all_products = set()
    for key, products in sugeset_product["user_products_based"].items():
        all_products.update(products)
    
    all_products.update(sugeset_product["similar_users_based"])
    unique_products = list(all_products)
    predictions = predication_rating(user_id, unique_products)
    sorted_ratings = dict(sorted(predictions.items(), key=lambda item: item[1], reverse=True))
    return sorted_ratings



#تابع لاستخراج معلومات المستخدم
def get_user_features(user_id):
    df = pd.read_csv(Config.DATA_PATHS['users'], low_memory=False)
    user_data = df[df['user_id'] == user_id]
    
    # إذا وجد المستخدم
    if not user_data.empty:
        # تحويل الصف الأول إلى قاموس
        user_feat = user_data.iloc[0].to_dict()
    else:
        return None
    user_features = {
        'user_id': user_feat['user_id'],
        'age': user_feat['age'],
        'gender': user_feat['gender'],
        'location_user': user_feat['location']
    }
    return user_features

# تابع لاستخراج معلومات المنتج
def get_product_features(product_id):
    df = pd.read_csv(Config.DATA_PATHS['products'], low_memory=False)
    pro_data = df[df['product_id'] == product_id]
    if not pro_data.empty:
        # تحويل الصف الأول إلى قاموس
        product_feat = pro_data.iloc[0].to_dict()
    else:
        return None
    product_features = {
        'product_id': product_feat['product_id'],
        'category': product_feat['category'],
        'price': product_feat['price'],
        'brand': product_feat['brand']
    }
    return product_features

def get_review_text(user_id,product_id):
    reviews_df = pd.read_csv(Config.DATA_PATHS['reviews'], low_memory=False)
    filtered = reviews_df[
        (reviews_df['user_id'] == user_id) & 
        (reviews_df['product_id'] == product_id)
    ]
    
    # إذا وجدت نتائج، نأخذ أول قيمة
    if not filtered.empty:
        return filtered.iloc[0]['review_text']
    else:
        return None 

# تابع يقوم بتوليد توصيات لكافة المستخدمين
def generate_recommendations_for_all_users():
  users_df = pd.read_csv(Config.DATA_PATHS['users'], low_memory=False)
  recommendations = []
  def get_recommendations(user_id):
    sugeset = suggest_for_user(user_id)
    all_products = set()
    for key, products in sugeset["user_products_based"].items():
        all_products.update(products)
    
    all_products.update(sugeset["similar_users_based"])
    unique_products = list(all_products)
    return [(user_id, product) for product in unique_products]
  all_recommendations = users_df['user_id'].apply(get_recommendations)
  recommendations_df = pd.DataFrame(
       [item for sublist in all_recommendations for item in sublist],
        columns=['user_id', 'recommended_product'])
  return recommendations_df


def cluster_all_users(users_df, reviews_df, interactions_df, n_clusters=5):
    # ===== 1. التحقق من البيانات ومعالجة القيم المفقودة =====
    users_df = users_df.copy()
    reviews_df = reviews_df.copy()
    interactions_df = interactions_df.copy()
    
    # معالجة القيم المفقودة في بيانات المستخدمين
    users_df.fillna({
        'age': users_df['age'].median(),
        'gender': 'unknown',
        'location': 'unknown'
    }, inplace=True)
    
    # ===== 2. تجهيز بيانات الصفات =====
    numeric_features = ['age']
    categorical_features = ['gender', 'location']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # تغيير مهم هنا
            ]), categorical_features)
        ])
    
    features_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('cluster', KMeans(n_clusters=n_clusters, random_state=42))
    ])
    
    # ===== 3. تجهيز بيانات التفاعلات =====
    interactions_df.fillna({'interaction_type': 'unknown'}, inplace=True)
    
    interaction_stats = interactions_df.groupby('user_id').agg({
        'interaction_type': [
            ('like_count', lambda x: (x == 'like').sum()),
            ('share_count', lambda x: (x == 'share').sum())
        ]
    }).reset_index()
    interaction_stats.columns = ['user_id', 'like_count', 'share_count']
    
    reviews_df.fillna({'rating': 0}, inplace=True)
    
    review_stats = reviews_df.groupby('user_id').agg({
        'rating': ['count', 'mean']
    }).reset_index()
    review_stats.columns = ['user_id', 'review_count', 'avg_rating']
    
    # دمج البيانات مع التأكد من وجود جميع المستخدمين
    all_users = pd.DataFrame({'user_id': users_df['user_id'].unique()})
    interaction_data = all_users.merge(
        interaction_stats.merge(
            review_stats,
            on='user_id',
            how='outer'
        ),
        on='user_id',
        how='left'
    ).fillna(0)
    
    # ===== 4. تطبيق التجميع =====
    # تجميع الصفات
    features_clusters = features_pipeline.fit_predict(users_df)
    
    # تجميع التفاعلات
    interaction_scaler = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler())
    ])
    
    interaction_scaled = interaction_scaler.fit_transform(
        interaction_data[['like_count', 'share_count', 'review_count', 'avg_rating']]
    )
    
    interaction_clusters = KMeans(
        n_clusters=n_clusters, 
        random_state=42
    ).fit_predict(interaction_scaled)
    
    # ===== 5. التجميع المدمج =====
    # التأكد من تطابق أحجام البيانات
    features_transformed = features_pipeline.named_steps['preprocessor'].transform(users_df)
    
    # الحصول على المستخدمين المشتركين
    common_users_mask = users_df['user_id'].isin(interaction_data['user_id']).values
    
    # تحويل المصفوفات إلى dense arrays إذا كانت sparse
    if hasattr(features_transformed, 'toarray'):
        features_transformed = features_transformed.toarray()
    
    # استخدام الفهرس المناسب للمصفوفات
    features_transformed = features_transformed[common_users_mask]
    interaction_scaled = interaction_scaled[common_users_mask]
    
    # التحقق من الأبعاد وإصلاحها
    print(f"features_transformed shape: {features_transformed.shape}")  # للتصحيح
    print(f"interaction_scaled shape: {interaction_scaled.shape}")      # للتصحيح
    
    # إذا كانت features_transformed أحادية البعد، نقوم بإعادة تشكيلها
    if features_transformed.ndim == 1:
        features_transformed = features_transformed.reshape(-1, 1)
    elif features_transformed.ndim > 2:
        features_transformed = features_transformed.reshape(features_transformed.shape[0], -1)
    
    if interaction_scaled.ndim == 1:
        interaction_scaled = interaction_scaled.reshape(-1, 1)
    elif interaction_scaled.ndim > 2:
        interaction_scaled = interaction_scaled.reshape(interaction_scaled.shape[0], -1)
    
    # التأكد من أن كلا المصفوفتين لهما نفس عدد الصفوف
    if features_transformed.shape[0] != interaction_scaled.shape[0]:
        min_rows = min(features_transformed.shape[0], interaction_scaled.shape[0])
        features_transformed = features_transformed[:min_rows]
        interaction_scaled = interaction_scaled[:min_rows]
    
    # الآن يمكننا دمج المصفوفتين
    combined_features = np.hstack([features_transformed, interaction_scaled])
    combined_features = np.nan_to_num(combined_features)
    
    combined_clusters = KMeans(
        n_clusters=n_clusters, 
        random_state=42
    ).fit_predict(combined_features)
    
    # ===== 6. بناء النتيجة النهائية =====
    result = {
        "feature_clusters": defaultdict(list),
        "interaction_clusters": defaultdict(list),
        "combined_clusters": defaultdict(list),
        "cluster_centers": {
            "features": features_pipeline.named_steps['cluster'].cluster_centers_,
            "interactions": KMeans(n_clusters=n_clusters, random_state=42).fit(interaction_scaled).cluster_centers_
        },
        "user_mapping": users_df[common_users_mask]['user_id'].tolist()[:features_transformed.shape[0]],
        "interaction_scaler_params": {
            'mean_': interaction_scaler.named_steps['scaler'].mean_,
            'scale_': interaction_scaler.named_steps['scaler'].scale_
        }
    }
    
    # تعبئة المجموعات مع مراعاة المستخدمين المشتركين فقط
    user_ids_common = users_df[common_users_mask]['user_id'].values[:features_transformed.shape[0]]
    interaction_user_ids_common = interaction_data[common_users_mask]['user_id'].values[:features_transformed.shape[0]]
    
    for idx, user_id in enumerate(user_ids_common):
        result["feature_clusters"][features_clusters[idx]].append(user_id)
        
    for idx, user_id in enumerate(interaction_user_ids_common):
        result["interaction_clusters"][interaction_clusters[idx]].append(user_id)
    
    for idx, user_id in enumerate(user_ids_common):
        result["combined_clusters"][combined_clusters[idx]].append(user_id)
    
    # 1. تجميع الصفات
    features_data = []
    for cluster_id, user_ids in result['feature_clusters'].items():
        for user_id in user_ids:
            features_data.append({
                'user_id': user_id,
                'feature_cluster': cluster_id
            })
    features_clusters_df = pd.DataFrame(features_data)
    
    # 2. تجميع التفاعلات
    interactions_data = []
    for cluster_id, user_ids in result['interaction_clusters'].items():
        for user_id in user_ids:
            interactions_data.append({
                'user_id': user_id,
                'interaction_cluster': cluster_id
            })
    interactions_clusters_df = pd.DataFrame(interactions_data)
    
    # 3. التجميع المدمج
    combined_data = []
    for cluster_id, user_ids in result['combined_clusters'].items():
        for user_id in user_ids:
            combined_data.append({
                'user_id': user_id,
                'combined_cluster': cluster_id
            })
    combined_clusters_df = pd.DataFrame(combined_data)
    
    # 4. مراكز المجموعات
    centers_data = []
    for cluster_type, centers in result['cluster_centers'].items():
        for cluster_id, center in enumerate(centers):
            centers_data.append({
                'cluster_type': cluster_type,
                'cluster_id': cluster_id,
                'center_values': center.tolist() if hasattr(center, 'tolist') else center
            })
    cluster_centers_df = pd.DataFrame(centers_data)
    
    return {
        'features_clusters': features_clusters_df,
        'interactions_clusters': interactions_clusters_df,
        'combined_clusters': combined_clusters_df,
        'cluster_centers': cluster_centers_df
    }

def suggest_products_by_description_from_user(
    products_df: pd.DataFrame,
    user_description: str,
    text_columns: List[str] = None,
    top_n: int = 5,
    similarity_threshold: float = 0.1,
    weights: Dict[str, float] = None,
    product_id_column: str = 'product_id'
) -> List[Union[str, int]]:
    """
    اقتراح منتجات بناء على وصف المستخدم وإرجاع قائمة بـ product_id فقط
    
    Parameters:
    -----------
    products_df : DataFrame
        هيكل بيانات المنتجات
    user_description : str
        الوصف الذي يدخله المستخدم
    text_columns : List[str], optional
        قائمة بالأعمدة النصية للبحث
    top_n : int, optional
        عدد المنتجات المراد اقتراحها (default: 5)
    similarity_threshold : float, optional
        الحد الأدنى لدرجة التشابه (default: 0.1)
    weights : Dict[str, float], optional
        أوزان للأعمدة المختلفة
    product_id_column : str, optional
        اسم عمود product_id (default: 'product_id')
    
    Returns:
    --------
    List[Union[str, int]]
        قائمة بـ product_id للمنتجات المقترحة
    """
    
    def clean_text(text: str) -> str:
        """تنظيف ومعالجة النص"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    # تنظيف وصف المستخدم
    cleaned_description = clean_text(user_description)
    if not cleaned_description:
        return []
    
    # التحقق من وجود عمود product_id
    if product_id_column not in products_df.columns:
        raise ValueError(f"عمود {product_id_column} غير موجود في DataFrame")
    
    # تحديد الأعمدة النصية تلقائياً إذا لم يتم تحديدها
    if text_columns is None:
        text_columns = []
        for col in products_df.columns:
            if products_df[col].dtype == 'object' and products_df[col].nunique() > 1:
                text_columns.append(col)
    
    if not text_columns:
        return []
    
    # أوزان افتراضية إذا لم يتم تحديدها
    if weights is None:
        weights = {col: 1.0/len(text_columns) for col in text_columns}
    
    # حساب التشابه لكل عمود
    total_similarities = np.zeros(len(products_df))
    
    for col in text_columns:
        if col not in products_df.columns:
            continue
            
        # إنشاء Vectorizer خاص بكل عمود
        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=1000
        )
        
        # تحضير بيانات العمود
        cleaned_col = products_df[col].fillna('').apply(clean_text)
        
        if cleaned_col.str.len().sum() == 0:
            continue
            
        try:
            # تدريب Vectorizer على عمود البيانات
            vectorizer.fit(cleaned_col)
            
            # تحويل وصف المستخدم
            user_vector = vectorizer.transform([cleaned_description])
            
            # تحويل عمود البيانات
            column_vectors = vectorizer.transform(cleaned_col)
            
            # حساب التشابه للعمود الحالي
            col_similarities = cosine_similarity(user_vector, column_vectors)[0]
            
            # إضافة إلى التشابه الكلي (مع الأوزان)
            weight = weights.get(col, 1.0/len(text_columns))
            total_similarities += col_similarities * weight
            
        except Exception as e:
            print(f"خطأ في معالجة العمود {col}: {e}")
            continue
    
    # إضافة التشابه الكلي إلى DataFrame مؤقت
    temp_df = products_df.copy()
    temp_df['similarity_score'] = total_similarities
    
    # ترتيب النتائج وتصفيتها
    filtered_results = temp_df[temp_df['similarity_score'] >= similarity_threshold]
    sorted_results = filtered_results.sort_values('similarity_score', ascending=False)
    
    # اختيار أفضل النتائج وإرجاع product_id فقط
    top_product_ids = sorted_results.head(top_n)[product_id_column].tolist()
    
    return top_product_ids

import os
import pickle
def get_cached_products_recommendations(product_ids):
        """الحصول على توصيات المنتجات من الكاش"""
        cache_path = Config.CACHE_PATH
        cache_file = os.path.join(cache_path, 'comprehensive_cache.pkl')
        if os.path.exists(cache_file):
                    with open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)
                    
                    product_cache = cache_data.get('product_cache', {})
        else:
            return
        results = defaultdict(list)
        
        for product_id in product_ids:
            if product_id in product_cache and 'error' not in product_cache[product_id]:
                product_data = product_cache[product_id]
                for key in product_data.keys():
                    results[key].extend(product_data[key])
            else:
                return
        
        # إزالة التكرارات
        for key in results:
            results[key] = list(set(results[key]))
        
        return dict(results)