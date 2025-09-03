import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import joblib
from tensorflow.keras.models import load_model
import logging
from recommendation_system.config.config import Config

class ImageBasedRecommender:
    def __init__(self, df, images_folder, image_column='images', batch_size=32):
        """
        Initialize the recommender system.
        
        Parameters:
        - df: DataFrame containing product information
        - images_folder: Path to the folder containing product images
        - image_column: Name of the column containing image filenames
        - batch_size: Batch size for processing images
        """
        self.df = df.copy()
        self.images_folder = images_folder
        self.image_column = image_column
        self.batch_size = batch_size
        self.model = self._initialize_model()
        self.image_features = None
        self.valid_indices = None
        self.logger = logging.getLogger(__name__)
        
        # Verify image files and filter dataframe
        self._verify_images()
        
        # Extract image features
        self._extract_features()
    
    def _initialize_model(self):
        """Initialize the ResNet50 model for feature extraction."""
        base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        return base_model
    
    def _verify_images(self):
        """Verify that all images in the dataframe exist in the images folder."""
        # print("Verifying image files...")
        
        # Check if images column exists
        if self.image_column not in self.df.columns:
            self.logger.error(f"Column '{self.image_column}' not found in dataframe")
            raise
        
        # Function to check if image exists
        def image_exists(img_name):
            if pd.isna(img_name):
                return False
            img_path = os.path.join(self.images_folder, img_name)
            return os.path.isfile(img_path)
        
        # Filter dataframe to only include products with valid images
        self.df['image_exists'] = self.df[self.image_column].apply(image_exists)
        valid_df = self.df[self.df['image_exists']].copy()
        
        if len(valid_df) == 0:
            self.logger.error(f"No valid images found in the specified folder")
            raise
        
        # print(f"Found {len(valid_df)}/{len(self.df)} products with valid images")
        self.df = valid_df
        self.valid_indices = self.df.index.tolist()
    
    def _load_and_preprocess_image(self, img_name):
        """Load and preprocess a single image."""
        img_path = os.path.join(self.images_folder, img_name)
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            return img_array
        except Exception as e:
            self.logger.error(f"Error processing image {img_name}: {str(e)}")
            return None
    
    def _extract_features(self):
        """Extract features for all valid images."""
        # print("Extracting image features...")
        
        image_names = self.df[self.image_column].tolist()
        features_list = []
        
        # Process images in batches
        for i in tqdm(range(0, len(image_names), self.batch_size)):
            batch_names = image_names[i:i + self.batch_size]
            batch_images = []
            
            for img_name in batch_names:
                img_array = self._load_and_preprocess_image(img_name)
                if img_array is not None:
                    batch_images.append(img_array)
            
            if batch_images:
                batch_images = np.vstack(batch_images)
                batch_features = self.model.predict(batch_images, verbose=0)
                features_list.append(batch_features)
        
        if features_list:
            self.image_features = np.vstack(features_list)
            # print(f"Successfully extracted features for {len(self.image_features)} images")
        else:
            self.logger.error(f"No features could be extracted from the images")
            raise 
    
    def recommend(self, query_image_name, top_n=5):
        """
        Recommend similar products based on image similarity.
        
        Parameters:
        - query_image_name: Name of the query image file
        - top_n: Number of recommendations to return
        
        Returns:
        - DataFrame with recommended products and their similarity scores
        """
        # Verify query image exists
        query_path = os.path.join(self.images_folder, query_image_name)
        if not os.path.isfile(query_path):
            self.logger.error(f"Query image '{query_image_name}' not found in images folder")
            raise 
        
        # Process query image
        query_img = self._load_and_preprocess_image(query_image_name)
        if query_img is None:
            self.logger.error(f"Could not process the query image")
            raise
        # Extract query features
        query_features = self.model.predict(query_img)
        
        # Calculate similarities
        similarities = cosine_similarity(query_features, self.image_features)[0]
        
        # Get top N recommendations (excluding the query image if it's in the dataset)
        similar_indices = np.argsort(similarities)[::-1]
        
        # Prepare results
        results = []
        for idx in similar_indices[:top_n + 1]:  # +1 in case the query is in the dataset
            product_id = self.df.iloc[idx]['_id']
            # Skip if this is the query product
            if self.df.iloc[idx][self.image_column] == query_image_name:
                continue
            results.append({
                'product_id': product_id,
                'image_name': self.df.iloc[idx][self.image_column],
                'similarity_score': similarities[idx]
            })
            if len(results) >= top_n:
                break
        
        return pd.DataFrame(results).head(top_n)


# إنشاء خدمة (Service) تدير النموذج والمتجهات

class ImageRecommendationService:
    # _instance = None
    def __init__(self, images_folder=Config.DATA_PATHS['image_folder']):
        self.images_folder = images_folder
        self.init_service()
 
    def init_service(self):
        """تهيئة الخدمة مرة واحدة عند التشغيل"""
        
        self.model = load_model(Config.DATA_PATHS['resnet50'])  # نموذج محفوظ مسبقاً
        self.features = joblib.load(Config.DATA_PATHS['image_features'])  # متجهات محفوظة
        self.df = pd.read_pickle(Config.DATA_PATHS['products_processed'])  # بيانات المنتجات
        
    def get_recommendations(self, image_name, top_n=5):
        """استرجاع التوصيات بسرعة"""
         # Verify query image exists
        query_path = os.path.join(self.images_folder, image_name)
        if not os.path.isfile(query_path):
            self.logger.error(f"Query image '{image_name}' not found in images folder")
            raise
        
        # Process query image
        query_img = self._load_and_preprocess_image(image_name)
        if query_img is None:
            self.logger.error(f"Could not process the query image")
            raise
        
        query_features = self.model.predict(query_img)
        
        # query_features = self._extract_features(image_name)
        similarities = cosine_similarity(query_features, self.features)[0]
        indices = np.argsort(similarities)[-top_n:][::-1]
        return self.df.iloc[indices].to_dict('records')
    
    def _load_and_preprocess_image(self, img_name):
        """Load and preprocess a single image."""
        img_path = os.path.join(self.images_folder, img_name)
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            return img_array
        except Exception as e:
            print(f"Error processing image {img_name}: {str(e)}")
            return None
    
# حفظ النموذج والمتجهات مسبقاً (مرة واحدة)
# خطوة تحضيرية (تجرى مرة واحدة)
def prepare_resources(df, image_path=Config.DATA_PATHS['image_folder']): 
    # 1. تهيئة النموذج الأصلي
    recommender = ImageBasedRecommender(df, image_path)
    
    # 2. حفظ النموذج
    recommender.model.save(Config.DATA_PATHS['resnet50'])
    
    # 3. حفظ متجهات السمات
    joblib.dump(recommender.image_features, Config.DATA_PATHS['image_features'])
    
    # 4. حفظ البيانات المعالجة
    df.to_pickle(Config.DATA_PATHS['products_processed'])


if __name__ == "__main__":
    # Load your dataframe
    # df = pd.read_csv('your_products.csv')
    
    # Initialize recommender
    # prepare_resources(df, image_path='data/image_data/images')
    
    # تهيئة الخدمة
    # service = ImageRecommendationService()
    # الحصول على توصيات
    # recommendations = service.get_recommendations("fa8e22d6-c0b6-5229-bb9e-ad52eda39a0a.jpg", 3)
    pass