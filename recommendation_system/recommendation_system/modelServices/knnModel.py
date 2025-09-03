from surprise import Dataset, Reader, KNNBasic
from surprise import accuracy
from surprise.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import logging
import os
from recommendation_system.config.config import Config


class KnnModel:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.data = self.load_and_preprocess_data()
        self.load_model()

    def load_and_preprocess_data(self):
        interactions = pd.read_csv(Config.DATA_PATHS['reviews']).dropna()
        
        # Filter users and products with few ratings
        user_rating_counts = interactions['user_id'].value_counts()
        item_rating_counts = interactions['product_id'].value_counts()
        
        # Keep users with at least 3 ratings
        interactions = interactions[interactions['user_id'].isin(
            user_rating_counts[user_rating_counts >= 3].index
        )]
        
        # Keep products with at least 3 ratings
        interactions = interactions[interactions['product_id'].isin(
            item_rating_counts[item_rating_counts >= 3].index
        )]

       
        return {'interactions': interactions}
    
    def train_knn(self):
        """Train KNN model"""
        try:
            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(
                self.data['interactions'][['user_id', 'product_id', 'rating']],
                reader
            )
            
            # Get KNN parameters from config
            sim_options = {
                'name': Config.MODEL_PARAMS['knn'].get('similarity', 'cosine'),
                'user_based': Config.MODEL_PARAMS['knn'].get('user_based', True),
                'k': Config.MODEL_PARAMS['knn'].get('k', 30),
                'min_k': Config.MODEL_PARAMS['knn'].get('min_k', 5)
            }
            
            self.model = KNNBasic(sim_options=sim_options)
            trainset = data.build_full_trainset()
            self.model.fit(trainset)
            
            # Save the model
            with open(Config.DATA_PATHS['knn_model'], 'wb') as f:
                pickle.dump(self.model, f)
                
        except Exception as e:
            self.logger.error(f"Error training KNN: {e}")
            raise
    
    def load_model(self):
        try:
            # Load KNN model
            if os.path.exists(Config.DATA_PATHS['knn_model']):
                with open(Config.DATA_PATHS['knn_model'], 'rb') as f:
                    self.model = pickle.load(f)
        
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")

    def recommend_with_knn(self, user_id, product_id):
        """Make recommendation using KNN model"""
        try:
            # Convert IDs to strings for compatibility
            user_id_str = str(user_id)
            product_id_str = str(product_id)
            
            interactions = self.data['interactions']
            
            # Calculate different averages
            global_avg = interactions['rating'].mean()
            user_avg = interactions[interactions['user_id'] == user_id]['rating'].mean()
            item_avg = interactions[interactions['product_id'] == product_id]['rating'].mean()
            
            # If user or item is completely new
            if pd.isna(user_avg) and pd.isna(item_avg):
                return global_avg
            
            # If user is new but item is known
            elif pd.isna(user_avg):
                return item_avg
            
            # If item is new but user is known
            elif pd.isna(item_avg):
                return user_avg
            
            # If both are known, use KNN prediction
            else:
                # Ensure model is trained
                if self.model is None:
                    self.train_knn()
                
                # Use KNN for prediction
                prediction = self.model.predict(user_id_str, product_id_str)
                return prediction.est
            
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return self.data['interactions']['rating'].mean()

    def evaluate_model(self, test_size=0.2, cv_folds=5):
        """
        Evaluate KNN model performance using various metrics
        
        Parameters:
        test_size: proportion of data to use for testing
        cv_folds: number of folds for cross-validation
        """
        try:
            # Load and prepare data
            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(
                self.data['interactions'][['user_id', 'product_id', 'rating']],
                reader
            )
            
            # Split data into train and test
            trainset, testset = train_test_split(data, test_size=test_size)
            
            # Get KNN parameters from config
            sim_options = {
                'name': Config.MODEL_PARAMS['knn'].get('similarity', 'cosine'),
                'user_based': Config.MODEL_PARAMS['knn'].get('user_based', True),
                'k': Config.MODEL_PARAMS['knn'].get('k', 30),
                'min_k': Config.MODEL_PARAMS['knn'].get('min_k', 5)
            }
            
            # Train model on training data
            model = KNNBasic(sim_options=sim_options)
            model.fit(trainset)
            
            # Predict on test data
            predictions = model.test(testset)
            
            # Calculate performance metrics
            rmse = accuracy.rmse(predictions, verbose=False)
            mae = accuracy.mae(predictions, verbose=False)
            
            # Calculate MSE manually for further analysis
            actual_ratings = [pred.r_ui for pred in predictions]
            predicted_ratings = [pred.est for pred in predictions]
            mse = mean_squared_error(actual_ratings, predicted_ratings)
            
            # Calculate prediction accuracy within a certain error margin
            error_margin = 1.0  # allowed error margin
            within_margin = sum(1 for i in range(len(actual_ratings)) 
                               if abs(actual_ratings[i] - predicted_ratings[i]) <= error_margin)
            accuracy_within_margin = within_margin / len(actual_ratings)
            
            # Error analysis by rating range
            error_analysis = self._analyze_errors_by_rating(actual_ratings, predicted_ratings)
            
            # Cross-validation
            cv_results = cross_validate(
                model, data, measures=['RMSE', 'MAE'], 
                cv=cv_folds, verbose=False
            )
            
            # Display results
            evaluation_results = {
                'RMSE': rmse,
                'MAE': mae,
                'MSE': mse,
                'Accuracy within ±1': accuracy_within_margin,
                'Cross Validation RMSE': np.mean(cv_results['test_rmse']),
                'Cross Validation MAE': np.mean(cv_results['test_mae']),
                'Error Analysis': error_analysis
            }
            
            # Plot evaluation graphs
            self._plot_evaluation_results(actual_ratings, predicted_ratings, evaluation_results)
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Error in model evaluation: {e}")
            raise
    
    def _analyze_errors_by_rating(self, actual_ratings, predicted_ratings):
        """Analyze errors by actual rating range"""
        error_analysis = {}
        
        # Group ratings by value
        rating_groups = {
            'Low (1-2)': [],
            'Medium (3)': [],
            'High (4-5)': []
        }
        
        for actual, predicted in zip(actual_ratings, predicted_ratings):
            error = abs(actual - predicted)
            
            if actual <= 2:
                rating_groups['Low (1-2)'].append(error)
            elif actual == 3:
                rating_groups['Medium (3)'].append(error)
            else:
                rating_groups['High (4-5)'].append(error)
        
        # Calculate mean error for each group
        for group, errors in rating_groups.items():
            if errors:
                error_analysis[group] = {
                    'Mean Error': np.mean(errors),
                    'Std Error': np.std(errors),
                    'Count': len(errors)
                }
        
        return error_analysis
    
    def _plot_evaluation_results(self, actual_ratings, predicted_ratings, evaluation_results):
        """Plot evaluation graphs for model performance"""
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('KNN Model Performance Evaluation', fontsize=16)
        
        # Plot 1: Predictions vs Actual Values
        axes[0, 0].scatter(actual_ratings, predicted_ratings, alpha=0.5)
        axes[0, 0].plot([1, 5], [1, 5], 'r--')  # Ideal line
        axes[0, 0].set_xlabel('Actual Ratings')
        axes[0, 0].set_ylabel('Predicted Ratings')
        axes[0, 0].set_title('Predictions vs. Actual Values')
        axes[0, 0].grid(True)
        
        # Plot 2: Error Distribution
        errors = [abs(a - p) for a, p in zip(actual_ratings, predicted_ratings)]
        axes[0, 1].hist(errors, bins=20, alpha=0.7, color='orange')
        axes[0, 1].axvline(np.mean(errors), color='red', linestyle='dashed', linewidth=1)
        axes[0, 1].set_xlabel('Error Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Prediction Errors')
        axes[0, 1].grid(True)
        
        # Plot 3: Performance Metrics Comparison
        metrics = ['RMSE', 'MAE', 'Accuracy within ±1']
        values = [evaluation_results[m] for m in metrics]
        axes[1, 0].bar(metrics, values, color=['blue', 'green', 'purple'])
        axes[1, 0].set_title('Model Performance Metrics')
        axes[1, 0].set_ylabel('Value')
        for i, v in enumerate(values):
            axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # Plot 4: Error Analysis by Rating Range
        error_analysis = evaluation_results['Error Analysis']
        groups = list(error_analysis.keys())
        mean_errors = [error_analysis[g]['Mean Error'] for g in groups]
        axes[1, 1].bar(groups, mean_errors, color=['red', 'blue', 'green'])
        axes[1, 1].set_title('Mean Error by Rating Range')
        axes[1, 1].set_ylabel('Mean Error')
        for i, v in enumerate(mean_errors):
            axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        plt.tight_layout()
        plt.savefig('knn_model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print detailed report
        print("=" * 50)
        print("KNN Model Evaluation Report")
        print("=" * 50)
        print(f"RMSE: {evaluation_results['RMSE']:.4f}")
        print(f"MAE: {evaluation_results['MAE']:.4f}")
        print(f"MSE: {evaluation_results['MSE']:.4f}")
        print(f"Accuracy within ±1: {evaluation_results['Accuracy within ±1']:.2%}")
        print(f"RMSE (Cross Validation): {evaluation_results['Cross Validation RMSE']:.4f}")
        print(f"MAE (Cross Validation): {evaluation_results['Cross Validation MAE']:.4f}")
        print("\nError Analysis by Rating Range:")
        for group, stats in evaluation_results['Error Analysis'].items():
            print(f"  {group}: Mean Error = {stats['Mean Error']:.3f}, Sample Count = {stats['Count']}")

    @staticmethod
    def under_sample_data(interactions, target_distribution=None):
        """Undersample dominant classes"""
        if target_distribution is None:
            target_distribution = {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2}
        
        balanced_data = pd.DataFrame()
        
        for rating in target_distribution.keys():
            rating_data = interactions[interactions['rating'] == rating]
            n_samples = int(len(interactions) * target_distribution[rating])
            
            if len(rating_data) > n_samples:
                # If samples are more than needed, take a random sample
                rating_data = rating_data.sample(n=n_samples, random_state=42)
            
            balanced_data = pd.concat([balanced_data, rating_data])
        
        return balanced_data

    @staticmethod
    def over_sample_data(interactions, target_distribution=None):
        """Oversample rare classes"""
        if target_distribution is None:
            # Make all classes equal
            max_count = interactions['rating'].value_counts().max()
            target_distribution = {rating: max_count for rating in range(1, 6)}
        
        balanced_data = pd.DataFrame()
        
        for rating in range(1, 6):
            rating_data = interactions[interactions['rating'] == rating]
            
            if len(rating_data) < target_distribution[rating]:
                # If samples are less than needed, repeat current samples
                n_samples_needed = target_distribution[rating] - len(rating_data)
                extra_samples = rating_data.sample(n=n_samples_needed, replace=True, random_state=42)
                rating_data = pd.concat([rating_data, extra_samples])
            
            balanced_data = pd.concat([balanced_data, rating_data])
        
        return balanced_data