import pandas as pd
from surprise import Dataset, Reader,SVD
import pickle
import logging
import os
from recommendation_system.config.config import Config

from surprise import accuracy
from surprise.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class SvdModel:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.data = self.load_and_preprocess_data()
        self.load_model()

    def load_and_preprocess_data(self):
            interactions = pd.read_csv(Config.DATA_PATHS['reviews']).dropna()
    
            # تصفية المستخدمين والمنتجات ذات التقييمات القليلة
            user_rating_counts = interactions['user_id'].value_counts()
            item_rating_counts = interactions['product_id'].value_counts()
            
            # الاحتفاظ بالمستخدمين الذين قدموا 3 تقييمات على الأقل
            interactions = interactions[interactions['user_id'].isin(
                user_rating_counts[user_rating_counts >= 3].index
            )]
            
            # الاحتفاظ بالمنتجات التي حصلت على 3 تقييمات على الأقل
            interactions = interactions[interactions['product_id'].isin(
                item_rating_counts[item_rating_counts >= 3].index
            )]
            
            interactions = interactions.drop_duplicates(subset=['user_id', 'product_id'])
            return {'interactions': interactions}
    
    def train_svd(self):
        """تدريب نموذج SVD"""
        try:
            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(
                self.data['interactions'][['user_id', 'product_id', 'rating']],
                reader
            )
            #  trainset = self.data['interactions'].build_full_trainset()
            self.model = SVD(**Config.MODEL_PARAMS['svd'])
            trainset = data.build_full_trainset()
            self.model.fit(trainset)
            
            # حفظ النموذج
            with open(Config.DATA_PATHS['svd_model'], 'wb') as f:
                pickle.dump(self.model, f)
                
        except Exception as e:
            self.logger.error(f"خطأ في تدريب SVD: {e}")
            raise
    
    def load_model(self):
        try:
            # تحميل نموذج SVD
            if os.path.exists(Config.DATA_PATHS['svd_model']):
                with open(Config.DATA_PATHS['svd_model'], 'rb') as f:
                    self.model = pickle.load(f)
        
        except Exception as e:
            self.logger.error(f"خطأ في تحميل النماذج: {e}")

    def recommend_with_svd(self, user_id, product_id):
            try:
                # تحويل المعرفات إلى strings للتأكد من التوافق
                user_id_str = str(user_id)
                product_id_str = str(product_id)
                
                interactions = self.data['interactions']
                
                # حساب المتوسطات المختلفة
                global_avg = interactions['rating'].mean()
                user_avg = interactions[interactions['user_id'] == user_id]['rating'].mean()
                item_avg = interactions[interactions['product_id'] == product_id]['rating'].mean()
                
                # إذا كان المستخدم أو المنتج جديداً تماماً
                if pd.isna(user_avg) and pd.isna(item_avg):
                    return global_avg
                
                # إذا كان المستخدم جديداً ولكن المنتج معروف
                elif pd.isna(user_avg):
                    return item_avg
                
                # إذا كان المنتج جديداً ولكن المستخدم معروف
                elif pd.isna(item_avg):
                    return user_avg
                
                # إذا كان كلاهما معروفين، استخدم تنبؤ SVD
                else:
                    # تأكد من أن النموذج مدرب
                    if self.model is None:
                        self.train_svd()
                    
                    # استخدم SVD للتنبؤ
                    prediction = self.model.predict(user_id_str, product_id_str)
                    return prediction.est
                
            except Exception as e:
                self.logger.error(f"خطأ في التنبؤ: {e}")
                return self.data['interactions']['rating'].mean()
    



    def evaluate_model(self, test_size=0.2, cv_folds=5):
        """
        تقييم أداء نموذج SVD باستخدام مقاييس متنوعة
        
        Parameters:
        test_size: نسبة البيانات المستخدمة للاختبار
        cv_folds: عدد الطيات للتحقق المتقاطع
        """
        try:
            # تحميل البيانات وإعدادها
            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(
                self.data['interactions'][['user_id', 'product_id', 'rating']],
                reader
            )
            
            # تقسيم البيانات إلى تدريب واختبار
            trainset, testset = train_test_split(data, test_size=test_size)
            
            # تدريب النموذج على بيانات التدريب
            model = SVD(**Config.MODEL_PARAMS['svd'])
            model.fit(trainset)
            
            # التنبؤ على بيانات الاختبار
            predictions = model.test(testset)
            
            # حساب مقاييس الأداء
            rmse = accuracy.rmse(predictions, verbose=False)
            mae = accuracy.mae(predictions, verbose=False)
            
            # حساب MSE يدوياً لمزيد من التحليل
            actual_ratings = [pred.r_ui for pred in predictions]
            predicted_ratings = [pred.est for pred in predictions]
            mse = mean_squared_error(actual_ratings, predicted_ratings)
            
            # حساب نسبة التنبؤات ضمن نطاق خطأ معين
            error_margin = 1.0  # نطاق الخطأ المسموح به
            within_margin = sum(1 for i in range(len(actual_ratings)) 
                               if abs(actual_ratings[i] - predicted_ratings[i]) <= error_margin)
            accuracy_within_margin = within_margin / len(actual_ratings)
            
            # تحليل الأخطاء حسب نطاق التقييم
            error_analysis = self._analyze_errors_by_rating(actual_ratings, predicted_ratings)
            
            # التحقق المتقاطع
            from surprise.model_selection import cross_validate
            cv_results = cross_validate(
                model, data, measures=['RMSE', 'MAE'], 
                cv=cv_folds, verbose=False
            )
            
            # عرض النتائج
            evaluation_results = {
                'RMSE': rmse,
                'MAE': mae,
                'MSE': mse,
                'Accuracy within ±1': accuracy_within_margin,
                'Cross Validation RMSE': np.mean(cv_results['test_rmse']),
                'Cross Validation MAE': np.mean(cv_results['test_mae']),
                'Error Analysis': error_analysis
            }
            
            # رسم رسوم بيانية للتقييم
            self._plot_evaluation_results(actual_ratings, predicted_ratings, evaluation_results)
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"خطأ في تقييم النموذج: {e}")
            raise
    
    def _analyze_errors_by_rating(self, actual_ratings, predicted_ratings):
        """تحليل الأخطاء حسب نطاق التقييم الفعلي"""
        error_analysis = {}
        
        # تجميع التقييمات حسب القيمة
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
        
        # حساب متوسط الخطأ لكل مجموعة
        for group, errors in rating_groups.items():
            if errors:
                error_analysis[group] = {
                    'Mean Error': np.mean(errors),
                    'Std Error': np.std(errors),
                    'Count': len(errors)
                }
        
        return error_analysis
    
    def _plot_evaluation_results(self, actual_ratings, predicted_ratings, evaluation_results):
        """رسم رسوم بيانية لتقييم أداء النموذج"""
        # إنشاء شكل برسوم بيانية متعددة
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('SVD Model Performance Evaluation', fontsize=16)
        
        # الرسم البياني 1: التنبؤات مقابل القيم الفعلية
        axes[0, 0].scatter(actual_ratings, predicted_ratings, alpha=0.5)
        axes[0, 0].plot([1, 5], [1, 5], 'r--')  # خط المثالية
        axes[0, 0].set_xlabel('Actual Ratings')
        axes[0, 0].set_ylabel('Expected Ratings')
        axes[0, 0].set_title('Predictions vs. Actual Values')
        axes[0, 0].grid(True)
        
        # الرسم البياني 2: توزيع الأخطاء
        errors = [abs(a - p) for a, p in zip(actual_ratings, predicted_ratings)]
        axes[0, 1].hist(errors, bins=20, alpha=0.7, color='orange')
        axes[0, 1].axvline(np.mean(errors), color='red', linestyle='dashed', linewidth=1)
        axes[0, 1].set_xlabel('Error value')
        axes[0, 1].set_ylabel('repetition')
        axes[0, 1].set_title('Distribution of prediction errors')
        axes[0, 1].grid(True)
        
        # الرسم البياني 3: مقارنة مقاييس الأداء
        metrics = ['RMSE', 'MAE', 'Accuracy within ±1']
        values = [evaluation_results[m] for m in metrics]
        axes[1, 0].bar(metrics, values, color=['blue', 'green', 'purple'])
        axes[1, 0].set_title('Model performance metrics')
        axes[1, 0].set_ylabel('value')
        for i, v in enumerate(values):
            axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # الرسم البياني 4: تحليل الأخطاء حسب نطاق التقييم
        error_analysis = evaluation_results['Error Analysis']
        groups = list(error_analysis.keys())
        mean_errors = [error_analysis[g]['Mean Error'] for g in groups]
        axes[1, 1].bar(groups, mean_errors, color=['red', 'blue', 'green'])
        axes[1, 1].set_title('Average error by evaluation range')
        axes[1, 1].set_ylabel('Average error')
        for i, v in enumerate(mean_errors):
            axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # طباعة تقرير مفصل
        print("=" * 50)
        print("SVD Model Evaluation Report")
        print("=" * 50)
        print(f"RMSE: {evaluation_results['RMSE']:.4f}")
        print(f"MAE: {evaluation_results['MAE']:.4f}")
        print(f"MSE: {evaluation_results['MSE']:.4f}")
        print(f"Accuracy within ±1: {evaluation_results['Accuracy within ±1']:.2%}")
        print(f"ross Validation RMSE: {evaluation_results['Cross Validation RMSE']:.4f}")
        print(f"Cross Validation MAE: {evaluation_results['Cross Validation MAE']:.4f}")
        print("\n Error analysis by evaluation range:")
        for group, stats in evaluation_results['Error Analysis'].items():
            print(f"  {group}: Mean Error= {stats['Mean Error']:.3f}, Count= {stats['Count']}")

