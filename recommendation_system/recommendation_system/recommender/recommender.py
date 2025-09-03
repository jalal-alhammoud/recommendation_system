from recommendation_system.functions.recommendingFuctions import mostPopular,extract_user_interests, cluster_all_users,get_user_interacted_products,get_high_rated_products,generate_recommendations_for_all_users
import pandas as pd
import logging
from recommendation_system.modelServices.svdModel import SvdModel
from recommendation_system.modelServices.knnModel import KnnModel
from recommendation_system.modelServices.deepLearnModel import DeepLearnModel
from recommendation_system.imageRecommender.imageRecommender import ImageRecommendationService, prepare_resources
from recommendation_system.config.config import Config

class Recommender:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.users = pd.read_csv(Config.DATA_PATHS['users'], low_memory=False)
        self.products = pd.read_csv(Config.DATA_PATHS['products'], low_memory=False)
        self.interactions = pd.read_csv(Config.DATA_PATHS['interactions'], low_memory=False)
        self.reviews = pd.read_csv(Config.DATA_PATHS['reviews'], low_memory=False)
        self.imageDf = pd.read_csv(Config.DATA_PATHS['images_df'])
        

    def save_most_popular(self):
        predication = mostPopular(self.reviews,1)
        predication = predication[['product_id', 'score', 'Rank']]
        predication.to_csv(Config.DATA_PATHS['most_popular'], index=False)

    
    def save_users_with_interests(self):
        users_with_interests = extract_user_interests(self.users, self.reviews, self.interactions ,self.products)
        users_with_interests['interacted_products'] = users_with_interests['user_id'].apply(
            lambda x: get_user_interacted_products(self.interactions, x))
        users_with_interests['high_rated_products'] = users_with_interests['user_id'].apply(
            lambda x: get_high_rated_products(self.reviews, x))
        users_with_interests.to_csv(Config.DATA_PATHS['users_with_interests'], index=False)

    
    def cluster_user(self):
        clusters = cluster_all_users(
            users_df=self.users,
            reviews_df=self.reviews,
            interactions_df=self.interactions,
            n_clusters=5  # عدد المجموعات
        )
        clusters['features_clusters'].to_csv(Config.DATA_PATHS['features_clusters'], index=False)
        clusters['interactions_clusters'].to_csv(Config.DATA_PATHS['interactions_clusters'], index=False)
        clusters['combined_clusters'].to_csv(Config.DATA_PATHS['combined_clusters'], index=False)
        clusters['cluster_centers'].to_csv(Config.DATA_PATHS['cluster_centers'], index=False)

    def generate_recommendations(self):
        recommendations_df = generate_recommendations_for_all_users()
        recommendations_df.to_csv(Config.DATA_PATHS['recommendations'], index=False)


    def init_train_models(self):
        svd= SvdModel()
        svd.train_svd()
        knn= KnnModel()
        knn.train_knn()
        deeplearn = DeepLearnModel()
        deeplearn.train_and_save_model()
        service = ImageRecommendationService()
        prepare_resources(self.imageDf, image_path=Config.DATA_PATHS['image_folder'])





    
