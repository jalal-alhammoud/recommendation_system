import pandas as pd
import numpy as np
import json
import pickle
from datetime import  timedelta
from datetime import datetime as dt
from collections import defaultdict
import os
import time
import threading
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')
from recommendation_system.functions.recommendingFuctions import *
from recommendation_system.config.config import Config

class UserRecommendationSystem:
    def __init__(self):
        self.cache_path = Config.CACHE_PATH
        self.data_paths = Config.DATA_PATHS
        self.user_cache = {}
        self.product_cache = {}
        self.cluster_cache = {}
        self.users_with_interests = None  
        self.user_clusters = {}  
        self.progress_file = os.path.join(self.cache_path, 'build_progress.json')
        self.lock = threading.Lock()
        os.makedirs(self.cache_path, exist_ok=True)
        self.product_clusters = {}
        self.user_interests = {}


    def load_all_data(self):
        """تحميل جميع البيانات مرة واحدة"""
        print("جاري تحميل البيانات...")
        self.df_user = pd.read_csv(self.data_paths['users'], low_memory=False)
        self.df_pro = pd.read_csv(self.data_paths['products'], low_memory=False)
        self.df_inter = pd.read_csv(self.data_paths['interactions'], low_memory=False)
        self.df_review = pd.read_csv(self.data_paths['reviews'], low_memory=False)
        self.pro_images = pd.read_csv(self.data_paths['images_df'], low_memory=False)
        
        # تحميل البيانات الأولية للمنتجات
        metadata = load_and_preprocess_data()
        self.df_metadata = metadata['products']
        pro_num = pd.DataFrame(metadata['features']['product_numerical'])
        pro_cat = pd.DataFrame(metadata['features']['product_categorical'])
        self.features = pd.concat([pro_num, pro_cat], axis=1)
        
        print("تم تحميل البيانات بنجاح")
    
    def load_build_progress(self):
        """تحميل تقدم البناء الحالي"""
        try:
            if os.path.exists(self.progress_file):
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"خطأ في تحميل تقدم البناء: {e}")
        return {
            'current_step': 0,
            'processed_users': [],
            'processed_products': [],
            'start_time': dt.now().isoformat(),
            'last_save_time': dt.now().isoformat(),
            'steps_completed': {
                'products_cache': False,
                'user_interests': False,
                'user_clusters': False,
                'user_recommendations': False
            }
        }
    
    def save_build_progress(self, progress_data):
        """حفظ تقدم البناء الحالي"""
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"خطأ في حفظ تقدم البناء: {e}")

    def save_partial_cache(self, cache_type, data):
        """حفظ جزء من الكاش"""
        try:
            cache_file = os.path.join(self.cache_path, f'partial_{cache_type}.pkl')
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"تم حفظ الكاش الجزئي لـ {cache_type}")
        except Exception as e:
            print(f"خطأ في حفظ الكاش الجزئي: {e}")

    def load_partial_cache(self, cache_type):
        """تحميل جزء من الكاش"""
        try:
            cache_file = os.path.join(self.cache_path, f'partial_{cache_type}.pkl')
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"خطأ في تحميل الكاش الجزئي: {e}")
        return {}
    

    def build_comprehensive_cache(self, resume=False, save_interval=300, batch_size=100):
        """بناء كاش شامل مع دعم الاستمرارية والحفظ الدوري"""
        print("بدء بناء الكاش الشامل...")
        
        progress = self.load_build_progress() if resume else {
            'current_step': 0,
            'processed_users': [],
            'processed_products': [],
            'start_time': (dt.now()).isoformat(),
            'last_save_time': dt.now().isoformat(),
            'steps_completed': {
                'products_cache': False,
                'user_interests': False,
                'user_clusters': False,
                'user_recommendations': False
            }
        }
        
        try:
            # الخطوة 1: بناء كاش المنتجات (فقط إذا لم يكن مكتملاً)
            if not progress['steps_completed']['products_cache']:
                print("بناء كاش المنتجات المتشابهة...")
                self.build_products_cache_with_progress(progress, save_interval, batch_size)
                progress['steps_completed']['products_cache'] = True
                self.save_build_progress(progress)
            
            # الخطوة 2: بناء كاش اهتمامات المستخدمين
            if not progress['steps_completed']['user_interests']:
                print("بناء كاش اهتمامات المستخدمين...")
                self.build_user_interests_cache()
                progress['steps_completed']['user_interests'] = True
                self.save_build_progress(progress)
            
            # الخطوة 3: بناء كاش تجميع المستخدمين
            if not progress['steps_completed']['user_clusters']:
                print("بناء كاش تجميع المستخدمين...")
                self.build_user_clusters_cache()
                progress['steps_completed']['user_clusters'] = True
                self.save_build_progress(progress)
            
            # الخطوة 4: بناء كاش التوصيات للمستخدمين
            if not progress['steps_completed']['user_recommendations']:
                print("بناء كاش توصيات المستخدمين...")
                self.build_user_recommendations_cache_with_progress(progress, save_interval, batch_size)
                progress['steps_completed']['user_recommendations'] = True
                self.save_build_progress(progress)
            
            # الخطوة 5: حفظ الكاش الشامل النهائي
            self.save_comprehensive_cache()
            
            # حذف ملف التقدم والكاش الجزئي بعد الانتهاء
            self.cleanup_temp_files()
            
            print("تم بناء الكاش الشامل بنجاح")
            
        except KeyboardInterrupt:
            print("تم إيقاف البناء يدوياً، سيتم حفظ التقدم الحالي...")
            self.save_build_progress(progress)
            self.save_partial_cache('user_cache', self.user_cache)
            self.save_partial_cache('product_cache', self.product_cache)
            print("تم حفظ التقدم الحالي، يمكن الاستئناف لاحقاً")
            
        except Exception as e:
            print(f"خطأ أثناء بناء الكاش: {e}")
            self.save_build_progress(progress)
            self.save_partial_cache('user_cache', self.user_cache)
            self.save_partial_cache('product_cache', self.product_cache)
            raise


    def build_products_cache_with_progress(self, progress, save_interval, batch_size):
        """بناء كاش المنتجات مع تتبع التقدم"""
        all_product_ids = self.df_pro['product_id'].unique().tolist()
        remaining_products = [pid for pid in all_product_ids if pid not in progress['processed_products']]
        
        start_time = dt.now()
        last_save_time = start_time
        
        for i, product_id in enumerate(remaining_products):
            try:
                product_id = int(product_id)

                recommendations = self.get_products_from_product([product_id])
                # print(product_id)
                print(recommendations)
                with self.lock:
                    self.product_cache[product_id] = recommendations
                    # print(product_id)
                    # print(recommendations)
                    progress['processed_products'].append(product_id)

                # الحفظ الدوري
                current_time = dt.now()
                if (current_time - last_save_time).total_seconds() >= save_interval:
                    self.save_build_progress(progress)
                    self.save_partial_cache('product_cache', self.product_cache)
                    last_save_time = current_time
                    print(f"تم حفظ التقدم المؤقت للمنتجات: {i+1}/{len(remaining_products)}")
                
                # طباعة التقدم كل 100 منتج
                if (i + 1) % 100 == 0:
                    elapsed = (current_time - start_time).total_seconds()
                    print(f"معالجة المنتجات: {i+1}/{len(remaining_products)} - الوقت المنقضي: {elapsed:.2f} ثانية")
                    
            except Exception as e:
                print(f"خطأ في المنتج {product_id}: {e}")
                with self.lock:
                    self.product_cache[product_id] = {'error': str(e)}
                    progress['processed_products'].append(product_id)
    

    def build_user_recommendations_cache_with_progress(self, progress, save_interval, batch_size):
        """بناء كاش التوصيات للمستخدمين مع تتبع التقدم"""
        if not self.product_cache:
            print("تحميل كاش المنتجات أولاً...")
            self.product_cache = self.load_partial_cache('product_cache') or {}
        all_user_ids = self.df_user['user_id'].unique().tolist()
        remaining_users = [uid for uid in all_user_ids if uid not in progress['processed_users']]
        
        start_time = dt.now()
        last_save_time = start_time
        
        for i, user_id in enumerate(remaining_users):
            try:
                recommendations = self.suggest_for_user(user_id)
                with self.lock:
                    self.user_cache[user_id] = {
                        'recommendations': recommendations,
                        'interests': self.get_user_interests(user_id),
                        'cluster_info': self.get_user_cluster_info(user_id),
                        'timestamp': dt.now().isoformat()
                    }
                    progress['processed_users'].append(user_id)
                
                # الحفظ الدوري
                current_time = dt.now()
                if (current_time - last_save_time).total_seconds() >= save_interval:
                    self.save_build_progress(progress)
                    self.save_partial_cache('user_cache', self.user_cache)
                    last_save_time = current_time
                    print(f"تم حفظ التقدم المؤقت للمستخدمين: {i+1}/{len(remaining_users)}")
                
                # طباعة التقدم كل 50 مستخدم
                if (i + 1) % 50 == 0:
                    elapsed = (current_time - start_time).total_seconds()
                    print(f"معالجة المستخدمين: {i+1}/{len(remaining_users)} - الوقت المنقضي: {elapsed:.2f} ثانية")
                    
            except Exception as e:
                print(f"خطأ في المستخدم {user_id}: {e}")
                with self.lock:
                    self.user_cache[user_id] = {'error': str(e)}
                    progress['processed_users'].append(user_id)
    


    def resume_build(self, save_interval=300):
        """استئناف بناء الكاش من النقطة التي توقف فيها"""
        print("محاولة استئناف بناء الكاش...")
        
        # تحميل التقدم الحالي
        progress = self.load_build_progress()
        
        # تحميل الكاش الجزئي إذا كان موجوداً
        partial_user_cache = self.load_partial_cache('user_cache')
        partial_product_cache = self.load_partial_cache('product_cache')
        
        if partial_user_cache:
            self.user_cache.update(partial_user_cache)
        if partial_product_cache:
            self.product_cache.update(partial_product_cache)
        
        # متابعة البناء
        self.build_comprehensive_cache(resume=True, save_interval=save_interval)

    def build_products_cache(self):
        """بناء كاش للمنتجات المتشابهة"""
        print("بناء كاش المنتجات المتشابهة...")
        all_product_ids = self.df_pro['product_id'].unique().tolist()
        
        for product_id in all_product_ids:
            try:
                recommendations = self.get_products_from_product([product_id])
                self.product_cache[product_id] = recommendations
            except Exception as e:
                print(f"خطأ في المنتج {product_id}: {e}")
                self.product_cache[product_id] = {'error': str(e)}
    
  

    def build_user_interests_cache(self):
        """بناء كاش لاهتمامات المستخدمين"""
        print("بناء كاش اهتمامات المستخدمين...")
        try:
            self.users_with_interests = self.extract_user_interests(
                self.df_user, self.df_review, self.df_inter, self.df_pro
            )
            print("تم بناء كاش اهتمامات المستخدمين بنجاح")
        except Exception as e:
            print(f"خطأ في بناء كاش اهتمامات المستخدمين: {e}")
            self.users_with_interests = None
    
   

    def build_user_clusters_cache(self):
        """بناء كاش لتجميع المستخدمين"""
        print("بناء كاش تجميع المستخدمين...")
        try:
            self.user_clusters = self.cluster_all_users(
                self.df_user, self.df_review, self.df_inter, n_clusters=5
            )
            print("تم بناء كاش تجميع المستخدمين بنجاح")
        except Exception as e:
            print(f"خطأ في بناء كاش تجميع المستخدمين: {e}")
            self.user_clusters = {}
    
    def build_user_recommendations_cache(self):
        """بناء كاش للتوصيات للمستخدمين"""
        print("بناء كاش توصيات المستخدمين...")
        all_user_ids = self.df_user['user_id'].unique().tolist()
        
        for user_id in all_user_ids:
            try:
                recommendations = self.suggest_for_user(user_id)
                self.user_cache[user_id] = {
                    'recommendations': recommendations,
                    'interests': self.get_user_interests(user_id),
                    'cluster_info': self.get_user_cluster_info(user_id),
                    'timestamp': dt.now().isoformat()
                }
            except Exception as e:
                print(f"خطأ في المستخدم {user_id}: {e}")
                self.user_cache[user_id] = {'error': str(e)}
    
    

    def get_user_interests(self, user_id):
            """الحصول على اهتمامات مستخدم معين"""
            if self.users_with_interests is None:
                return {}
            
            user_data = self.users_with_interests[self.users_with_interests['user_id'] == user_id]
            if not user_data.empty:
                return user_data.iloc[0].to_dict()
            return {}
    
   
    def get_user_cluster_info(self, user_id):
            """الحصول على معلومات تجميع مستخدم معين"""
            cluster_info = {}
            if not self.user_clusters:
                return cluster_info
            
            for cluster_type in ['features_clusters', 'interactions_clusters', 'combined_clusters']:
                if cluster_type in self.user_clusters:
                    cluster_df = self.user_clusters[cluster_type]
                    user_cluster = cluster_df[cluster_df['user_id'] == user_id]
                    if not user_cluster.empty:
                        cluster_info[cluster_type] = user_cluster.iloc[0][f"{cluster_type.split('_')[0]}_cluster"]
            return cluster_info
    
    
    def save_comprehensive_cache(self):
        cache_data = {
            'user_cache': self.user_cache,
            'product_cache': self.product_cache,
            'user_clusters': self.user_clusters,
            'product_clusters': self.product_clusters,
            'user_interests': self.user_interests,
            # 'last_build_time': self.last_build_time
        }
        
        # تحويل numpy types إلى Python native types
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        cache_data = convert_to_serializable(cache_data)
        
    
        with open(f'{self.cache_path}/comprehensive_cache.json', 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
    def cleanup_temp_files(self):
        """حذف الملفات المؤقتة بعد الانتهاء"""
        try:
            if os.path.exists(self.progress_file):
                os.remove(self.progress_file)
            
            partial_files = [
                os.path.join(self.cache_path, 'partial_user_cache.pkl'),
                os.path.join(self.cache_path, 'partial_product_cache.pkl')
            ]
            
            for file_path in partial_files:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    
            print("تم حذف الملفات المؤقتة")
        except Exception as e:
            print(f"خطأ في حذف الملفات المؤقتة: {e}")


  
    def load_comprehensive_cache(self):
            """تحميل الكاش الشامل مع دعم الاستمرارية"""
            try:
                cache_file = os.path.join(self.cache_path, 'comprehensive_cache.pkl')
                if os.path.exists(cache_file):
                    with open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)
                    
                    self.product_cache = cache_data.get('product_cache', {})
                    self.user_cache = cache_data.get('user_cache', {})
                    
                    # تحميل البيانات الاختيارية
                    if 'user_interests' in cache_data:
                        self.users_with_interests = pd.DataFrame(cache_data['user_interests'])
                    
                    if 'user_clusters' in cache_data:
                        self.user_clusters = {}
                        for key, value in cache_data['user_clusters'].items():
                            self.user_clusters[key] = pd.DataFrame(value)
                    
                    print("تم تحميل الكاش الشامل بنجاح")
                    return True
                
                print("لم يتم العثور على كاش كامل")
                return False
        
            except Exception as e:
                print(f"خطأ في تحميل الكاش: {e}")
                self.user_cache = {}
                self.product_cache = {}
                self.user_clusters = {}
                self.product_clusters = {}  
                self.user_interests = {}
                self.last_build_time = 0
                return False
    
    def get_build_status(self):
        """الحصول على حالة بناء الكاش"""
        progress = self.load_build_progress()
        
        if not progress:
            return {"status": "not_started", "message": "لم يبدأ بناء الكاش بعد"}
        
        total_users = len(self.df_user['user_id'].unique()) if hasattr(self, 'df_user') else 0
        total_products = len(self.df_pro['product_id'].unique()) if hasattr(self, 'df_pro') else 0
        
        processed_users = len(progress.get('processed_users', []))
        processed_products = len(progress.get('processed_products', []))
        
        status = {
            "status": "in_progress" if not all(progress['steps_completed'].values()) else "completed",
            "start_time": progress.get('start_time'),
            "last_save_time": progress.get('last_save_time'),
            "steps_completed": progress['steps_completed'],
            "progress": {
                "users": f"{processed_users}/{total_users}",
                "products": f"{processed_products}/{total_products}",
                "users_percentage": (processed_users / total_users * 100) if total_users > 0 else 0,
                "products_percentage": (processed_products / total_products * 100) if total_products > 0 else 0
            }
        }
        
        return status
    
    def build_in_background(self, save_interval=300, batch_size=100):
        """بناء الكاش في الخلفية"""
        def build_thread():
            try:
                self.build_comprehensive_cache(save_interval=save_interval, batch_size=batch_size)
            except Exception as e:
                print(f"خطأ في بناء الكاش في الخلفية: {e}")
        
        thread = threading.Thread(target=build_thread, daemon=True)
        thread.start()
        return thread
    # ===== التوابع الأساسية (معدلة للاستفادة من الكاش) =====
    
    def get_cached_products_recommendations(self, product_ids):
        """الحصول على توصيات المنتجات من الكاش"""
        results = defaultdict(list)
        
        for product_id in product_ids:
            if product_id in self.product_cache and 'error' not in self.product_cache[product_id]:
                product_data = self.product_cache[product_id]
                for key in product_data.keys():
                    results[key].extend(product_data[key])
        
        # إزالة التكرارات
        for key in results:
            results[key] = list(set(results[key]))
        
        return dict(results)
    
    def get_cached_user_recommendations(self, user_id):
        """الحصول على توصيات المستخدم من الكاش"""
        if user_id in self.user_cache and 'error' not in self.user_cache[user_id]:
            return self.user_cache[user_id]['recommendations']
        
        # إذا لم يكن في الكاش، حساب التوصيات مباشرة
        return self.suggest_for_user(user_id)
    
    def get_user_analytics(self, user_id):
        """الحصول على تحليلات شاملة للمستخدم"""
        if user_id not in self.user_cache:
            return {"error": "User not found in cache"}
        
        user_data = self.user_cache[user_id]
        if 'error' in user_data:
            return user_data
        
        return {
            'basic_info': self.df_user[self.df_user['user_id'] == user_id].iloc[0].to_dict(),
            'recommendations': user_data['recommendations'],
            'interests': user_data['interests'],
            'cluster_info': user_data['cluster_info'],
            'interaction_stats': self.get_user_interaction_stats(user_id),
            'review_stats': self.get_user_review_stats(user_id)
        }
    
    def get_user_interaction_stats(self, user_id):
        """إحصائيات تفاعلات المستخدم"""
        user_interactions = self.df_inter[self.df_inter['user_id'] == user_id]
        return {
            'total_interactions': len(user_interactions),
            'interaction_types': user_interactions['interaction_type'].value_counts().to_dict(),
            'last_interaction': user_interactions['timestamp'].max() if not user_interactions.empty else None
        }
    
    def get_user_review_stats(self, user_id):
        """إحصائيات تقييمات المستخدم"""
        user_reviews = self.df_review[self.df_review['user_id'] == user_id]
        return {
            'total_reviews': len(user_reviews),
            'average_rating': user_reviews['rating'].mean() if not user_reviews.empty else 0,
            'rating_distribution': user_reviews['rating'].value_counts().to_dict()
        }
    
    def get_cluster_analytics(self, cluster_type='combined'):
        """تحليلات المجموعات"""
        cluster_df = self.user_clusters.get(f'{cluster_type}_clusters')
        if cluster_df is None:
            return {"error": "Cluster type not found"}
        
        analytics = {}
        for cluster_id in cluster_df[f'{cluster_type}_cluster'].unique():
            cluster_users = cluster_df[cluster_df[f'{cluster_type}_cluster'] == cluster_id]['user_id']
            
            # إحصائيات المستخدمين في المجموعة
            users_in_cluster = self.df_user[self.df_user['user_id'].isin(cluster_users)]
            
            analytics[cluster_id] = {
                'user_count': len(cluster_users),
                'avg_age': users_in_cluster['age'].mean(),
                'gender_distribution': users_in_cluster['gender'].value_counts().to_dict(),
                'location_distribution': users_in_cluster['location'].value_counts().to_dict()
            }
        
        return analytics
    
    # ===== واجهة برمجية للاستخدام في الموقع =====
    
    def api_get_recommendations(self, user_id):
        """واجهة برمجية للحصول على توصيات للمستخدم"""
        return self.get_cached_user_recommendations(user_id)
    
    def api_get_user_analytics(self, user_id):
        """واجهة برمجية للحصول على تحليلات المستخدم"""
        return self.get_user_analytics(user_id)
    
    def api_get_similar_products(self, product_ids):
        """واجهة برمجية للحصول على منتجات مشابهة"""
        return self.get_cached_products_recommendations(product_ids)
    
    def api_get_cluster_info(self, cluster_type='combined'):
        """واجهة برمجية للحصول على معلومات المجموعات"""
        return self.get_cluster_analytics(cluster_type)
    
    # ===== التوابع المساعدة (يجب تنفيذها) =====
    
    def get_products_from_product(self, product_ids):
        """تابع الحصول على منتجات مشابهة (مطلوب تنفيذه)"""
        return get_products_from_product(product_ids)
       
    
    def extract_user_interests(self, users_df, reviews_df, interactions_df, products_df):
        """تابع استخراج اهتمامات المستخدمين (مطلوب تنفيذه)"""
        return extract_user_interests(users_df, reviews_df, interactions_df, products_df)
        
    
    def cluster_all_users(self, users_df, reviews_df, interactions_df, n_clusters=5):
        """تابع تجميع المستخدمين (مطلوب تنفيذه)"""
        return cluster_all_users(users_df, reviews_df, interactions_df, n_clusters=5)
       
    
    def suggest_for_user(self, user_id):
        """تابع اقتراح التوصيات للمستخدم (مطلوب تنفيذه)"""
        self.df_user = pd.read_csv(self.data_paths['users'], low_memory=False)
        self.df_pro = pd.read_csv(self.data_paths['products'], low_memory=False)
        self.df_inter = pd.read_csv(self.data_paths['interactions'], low_memory=False)
        self.df_review = pd.read_csv(self.data_paths['reviews'], low_memory=False)
        #فحص لمعرفة إذا كان المستخدم قد تفاعل سابقاً مع الموقع
        if  (user_id in self.df_inter['user_id'].values) :
            user_product = []
            user_product = get_user_interacted_products(self.df_inter, user_id)
            if (user_id in self.df_review['user_id'].values):
                user_product = user_product +  get_high_rated_products(self.df_review, user_id)
            # user_products_based= get_products_from_product(user_product)
            user_products_based = {}
            for product_id in user_product:
                if product_id in self.product_cache and 'error' not in self.product_cache[product_id]:
                    # دمج نتائج الكاش للمنتجات المختلفة
                    for key, value in self.product_cache[product_id].items():
                        if key not in user_products_based:
                            user_products_based[key] = []
                        user_products_based[key].extend(value)
            
            # إزالة التكرارات من النتائج
            for key in user_products_based:
                user_products_based[key] = list(set(user_products_based[key]))
        else:
            most_popular = mostPopular(self.df_review,11)
            list_most_popular =  most_popular['product_id'].tolist()
            user_products_based ={'most_popular': list_most_popular} 
        similar_users_based= get_user_similar_prefered_products(user_id, self.df_user, self.df_pro,self.df_inter, self.df_review )
        recommendation = {
        "user_products_based": user_products_based,
        "similar_users_based": similar_users_based
        }
        return recommendation

# ===== الاستخدام =====

if __name__ == "__main__":
     # تهيئة النظام
    recommendation_system = UserRecommendationSystem()
    
    # تحميل البيانات
    recommendation_system.load_all_data()
    
    # خيارات البناء:
    # 1. بناء جديد
    # recommendation_system.build_comprehensive_cache(save_interval=300, batch_size=100)
    
    # 2. استئناف بناء موجود
    # recommendation_system.resume_build(save_interval=300)
    
    # 3. البناء في الخلفية
    # thread = recommendation_system.build_in_background(save_interval=300, batch_size=100)
    # print("جاري بناء الكاش في الخلفية...")
    # thread.join()  # الانتظار حتى الانتهاء إذا أردت
    
    # 4. التحقق من الحالة
    status = recommendation_system.get_build_status()
    print("حالة بناء الكاش:", status)
    
    # أو تحميل الكاش الموجود
    if recommendation_system.load_comprehensive_cache():
        # أمثلة للاستخدام
        user_id = 123
        
        # الحصول على توصيات
        recommendations = recommendation_system.api_get_recommendations(user_id)
        print("التوصيات:", recommendations)