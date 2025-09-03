import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Config:
    CACHE_PATH = os.path.join(BASE_DIR, 'recommendation_system','cash_data')
    # إعدادات النماذج
    MODEL_PARAMS = {
        'svd': {
           'n_factors': 50,      
            'n_epochs': 50,       
            'lr_all': 0.007,      
            'reg_all':0.01,      
            'init_mean': 0,       
            'init_std_dev': 0.05, 
        },
        'knnWithMeans': {
        'k': 20,  # عدد الجيران المستخدمين
        'sim_options': {
            'name': 'pearson',  # مقياس التشابه (يمكن أن يكون 'cosine', 'msd', 'pearson')
            'user_based': True , # True للتصفية القائمة على المستخدم، False للتصفية القائمة على العنصر
            'min_support': 3  # الحد الأدنى للعناصر المشتركة
        },
        'verbose': False
    },
    'knn':{
        'similarity': 'msd',
        'user_based': False,
        'k': 40,
        'min_k': 5
    },

        'lightfm': {
            'no_components': 30,
            'loss': 'warp',
            'learning_rate': 0.05,
            'item_alpha': 0.001
        },
        'hybrid': {
            'user_dim': 15,  
            'product_dim': 50,  
            'hidden_units': 256
        }
    }
    
    # مسارات الملفات
    DATA_PATHS = {
                # ملفات البيانات الأساسية
            'users': os.path.join(BASE_DIR, 'data', 'users.csv'),
            'products': os.path.join(BASE_DIR, 'data', 'products.csv'),
            'reviews': os.path.join(BASE_DIR, 'data', 'reviews.csv'),
            'interactions': os.path.join(BASE_DIR, 'data', 'interactions.csv'),
            'context': os.path.join(BASE_DIR, 'data', 'context.csv'),
            'training_data_merged': os.path.join(BASE_DIR, 'data', 'training_data_merged.csv'),
            
            # مسارات متعلقة بالصور
            'images_df': os.path.join(BASE_DIR, 'data','images_data.csv'),
            'image_folder': os.path.join(BASE_DIR, 'static', 'uploads', 'products'),
            
            # نماذج التعلم الآلي
            'resnet50': os.path.join(BASE_DIR,'recommendation_system' ,'models', 'resnet50.h5'),
            'hybrid_model': os.path.join(BASE_DIR,'recommendation_system' ,'models', 'hybrid_model.keras'),
            'knn_model': os.path.join(BASE_DIR,'recommendation_system','models', 'knn_model.pkl'),
            'lightfm_model': os.path.join(BASE_DIR,'recommendation_system' ,'models', 'lightfm_model.pkl'),
            'svd_model': os.path.join(BASE_DIR,'recommendation_system' ,'models', 'svd_model.pkl'),
            'knnWithMeans_model': os.path.join(BASE_DIR,'recommendation_system' ,'models', 'KnnWithMeans.pkl'),
            
            # ملفات البيانات المعالجة
            'image_features': os.path.join(BASE_DIR,'recommendation_system' ,'models', 'data', 'image_features.pkl'),
            'products_processed': os.path.join(BASE_DIR,'recommendation_system' ,'models', 'data', 'products_processed.pkl'),
            
            # ملفات نظام التوصية
            'most_popular': os.path.join(BASE_DIR,'recommendation_system' ,'recommender', 'recommender_data', 'most_popular.csv'),
            'users_with_interests': os.path.join(BASE_DIR,'recommendation_system' ,'recommender', 'recommender_data', 'users_with_interests.csv'),
            'features_clusters': os.path.join(BASE_DIR,'recommendation_system' ,'recommender', 'recommender_data', 'features_clusters.csv'),
            'interactions_clusters': os.path.join(BASE_DIR,'recommendation_system' ,'recommender', 'recommender_data', 'interactions_clusters.csv'),
            'combined_clusters': os.path.join(BASE_DIR,'recommendation_system' ,'recommender', 'recommender_data', 'combined_clusters.csv'),
            'cluster_centers': os.path.join(BASE_DIR,'recommendation_system' ,'recommender', 'recommender_data', 'cluster_centers.csv'),
            'recommendations': os.path.join(BASE_DIR,'recommendation_system' ,'recommender', 'recommender_data', 'recommendations.csv')

    }

    

    