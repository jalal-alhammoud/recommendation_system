import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.losses import Huber
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score,classification_report,confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from imblearn.over_sampling import SMOTE
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from recommendation_system.config.config import Config
from recommendation_system.modelServices.data_processor import DataProcessor

class DeepLearnModel:
    def __init__(self):
        self.model_path = Config.DATA_PATHS['hybrid_model']
        self.user_mapping = None
        self.product_mapping = None
        self.rf_model = None
        self.model, self.processor = self.load_hybrid_model()
       
        self.ensemble_weights = [0.5, 0.5]

    def preprocess_data(self):
        users = pd.read_csv(Config.DATA_PATHS['users'])
        products = pd.read_csv(Config.DATA_PATHS['products'])
        reviews_df = pd.read_csv(Config.DATA_PATHS['reviews'])

        # تصفية المستخدمين والمنتجات ذات التقييمات القليلة
        user_rating_counts = reviews_df['user_id'].value_counts()
        item_rating_counts = reviews_df['product_id'].value_counts()
        
        reviews_df = reviews_df[reviews_df['user_id'].isin(
            user_rating_counts[user_rating_counts >= 3].index
        )]
        
        reviews_df = reviews_df[reviews_df['product_id'].isin(
            item_rating_counts[item_rating_counts >= 3].index
        )]

        reviews = reviews_df.drop_duplicates(subset=['user_id', 'product_id'])
        products = products[['product_id', 'price', 'category', 'brand']]
        users = users.rename(columns={'location': 'location_user'})
        
        merged_data = pd.merge(reviews, users, on='user_id')
        merged_data = pd.merge(merged_data, products, on='product_id')
        
        # معالجة القيم المفقودة
        merged_data['rating'] = merged_data['rating'].fillna(3)
        merged_data = merged_data[['user_id','product_id','age', 'gender', 'location_user', 'category', 'price', 'brand','rating']]
        # موازنة البيانات باستخدام التقنية المحسنة
        # merged_data = self.improved_balance_data(merged_data, 'rating')
        
        # حفظ المفاتيح الأصلية
        self.user_mapping = {user_id: idx for idx, user_id in enumerate(merged_data['user_id'].unique())}
        self.product_mapping = {product_id: idx for idx, product_id in enumerate(merged_data['product_id'].unique())}
        
        # ترميز المعرفات
        merged_data['user_id_encoded'] = merged_data['user_id'].map(self.user_mapping)
        merged_data['product_id_encoded'] = merged_data['product_id'].map(self.product_mapping)
        
        processor = DataProcessor()
        processor.fit(merged_data)
        
        processor_path = self.model_path.replace('.keras', '_processor.keras')
        with open(processor_path, 'wb') as f:
            pickle.dump(processor, f)
        
        # تحويل السمات الفئوية
        cat_cols = ['gender', 'location_user', 'category', 'brand']
        for col in cat_cols:
            if col in merged_data.columns:
                merged_data[col] = LabelEncoder().fit_transform(merged_data[col].astype(str))

        # تطبيع السمات الرقمية
        num_cols = ['age', 'price']
        scaler = StandardScaler()
        merged_data[num_cols] = scaler.fit_transform(merged_data[num_cols])

        # حفظ بيانات التدريب
        training_data_path = Config.DATA_PATHS['training_data_merged']
        merged_data.to_csv(training_data_path, index=False)
        
        return merged_data
    
    def load_hybrid_model(self):
        try:
            custom_objects = {
                'SafeCastLayer': SafeCastLayer,
                'SafeClipLayer': SafeClipLayer,
                'Adam': Adam
            }
            
            model = keras.models.load_model(
                self.model_path, 
                compile=False,
                custom_objects=custom_objects,
                safe_mode=False
            )
            model.compile(
                # optimizer=Adam(learning_rate=0.001),
                # loss='mse',
                # metrics=['mae']
                 optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'sparse_categorical_accuracy']
                
            )

            processor_path = self.model_path.replace('.keras', '_processor.keras')
            if os.path.exists(processor_path):
                with open(processor_path, 'rb') as f:
                    processor = pickle.load(f)
            else:
                processor = None

            user_mapping_path= self.model_path.replace('.keras', '_user_mapping.pkl')
            with open(user_mapping_path, 'rb') as f:
                 self.user_mapping = pickle.load(f)

            product_mapping_path= self.model_path.replace('.keras', '_product_mapping.pkl')
        
            with open(product_mapping_path, 'rb') as f:
                self.product_mapping = pickle.load(f)


            return model, processor

        except Exception as e:
            print(f"فشل في التحميل: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

    def improved_balance_data(self, df, target_column):
        """تحسين موازنة البيانات باستخدام SMOTE"""
        rating_counts = df[target_column].value_counts()
        print("توزيع التقييمات الأصلي:", rating_counts)
        
        # فصل الميزات والهدف
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        # تطبيق SMOTE
        smote = SMOTE(random_state=42, k_neighbors=2)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # إعادة إنشاء DataFrame
        balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
        balanced_df[target_column] = y_resampled
        
        print("توزيع التقييمات بعد الموازنة:", balanced_df[target_column].value_counts())
        
        return balanced_df

    def train_and_save_model(self):
        data = self.preprocess_data()
        
        # تدريب Random Forest أولاً للحصول على insights
        self.rf_model = self.train_random_forest(data)
        
        # تحسين النموذج العميق بناءً على insights من Random Forest
        model, history = self.enhanced_training_with_insights(data, self.rf_model)
        
        # حفظ النموذج
        model.save(self.model_path)
        

        user_mapping_path= self.model_path.replace('.keras', '_user_mapping.pkl')
        with open(user_mapping_path, 'wb') as f:
                 pickle.dump(self.user_mapping, f)

        product_mapping_path= self.model_path.replace('.keras', '_product_mapping.pkl')
    
        with open(product_mapping_path, 'wb') as f:
            pickle.dump(self.product_mapping, f)
       
        return model, history
    
    def train_random_forest(self, data):
        """تدريب نموذج Random Forest للحصول على insights"""
        X = data[['age', 'gender', 'location_user', 'category', 'price', 'brand']]
        y = data['rating']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # نموذج Random Forest مع معلمات محسنة
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        rf_model.fit(X_train, y_train)
        
        # تقييم الأداء
        predictions = rf_model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        
        print(f" Random Forest MAE: {mae}")
        print(f" Random Forest RMSE: {rmse}")
        
        return rf_model

    def feature_importance_analysis(self, data, rf_model):
        """تحليل أهمية الميزات من Random Forest"""
        feature_names = ['age', 'gender', 'location_user', 'category', 'price', 'brand']
        importances = rf_model.feature_importances_
        
        # عرض أهمية الميزات
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(" أهمية الميزات من Random Forest:")
        print(feature_importance_df)
        
        return feature_importance_df

    def create_feature_optimized_model(self, num_users, num_products, feature_importance_df):
        """نموذج يستخدم insights من أهمية الميزات"""
        # الميزات الأكثر أهمية
        top_features = feature_importance_df.head(3)['feature'].tolist()
        print(f"الميزات الأكثر أهمية: {top_features}")
        
        # مدخلات النموذج
        user_id_input = Input(shape=(1,), name='user_id')
        product_id_input = Input(shape=(1,), name='product_id')
        
        # مدخل للميزات المهمة
        important_features_input = Input(shape=(3,), name='important_features')
       

        user_embedding = Embedding(num_users + 1, 64, name='user_embedding')(user_id_input)
        user_embedding = Flatten()(user_embedding)

        product_embedding = Embedding(num_products + 1, 64, name='product_embedding')(product_id_input)
        product_embedding = Flatten()(product_embedding)
        # معالجة الميزات المهمة
        features_dense = Dense(32, activation='relu')(important_features_input)
        
        # دمج المميزات
        concat = Concatenate()([
            user_embedding, product_embedding, features_dense
        ])

        dense = Dense(128, activation='relu')(concat)
        dense = BatchNormalization()(dense)
        dense = Dropout(0.4)(dense)
        
        dense = Dense(64, activation='relu')(dense)
        dense = BatchNormalization()(dense)
        dense = Dropout(0.3)(dense)

        dense = Dense(64, activation='relu')(dense)
        dense = BatchNormalization()(dense)
        dense = Dropout(0.3)(dense)

        dense = Dense(32, activation='relu')(dense)
        dense = BatchNormalization()(dense)
        dense = Dropout(0.3)(dense)

        dense = Dense(16, activation='relu')(dense)
        dense = BatchNormalization()(dense)
        dense = Dropout(0.3)(dense)

        dense = Dense(8, activation='relu')(dense)
        dense = BatchNormalization()(dense)
        dense = Dropout(0.3)(dense)
        
        
        output = Dense(6, activation='softmax')(dense)  # 6 فئات (0-5)
        

        model = Model(
            inputs=[
                user_id_input, product_id_input,
                important_features_input
            ],
            outputs=output
        )

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'sparse_categorical_accuracy']
            # loss='mse',
            # metrics=['mae']
        )
        
        return model

    def enhanced_training_with_insights(self, data, rf_model):
        """تدريب محسن باستخدام insights من Random Forest"""
        # تحليل أهمية الميزات
        feature_importance_df = self.feature_importance_analysis(data, rf_model)
        
        # إعداد البيانات مع التركيز على الميزات المهمة
        top_features = feature_importance_df.head(3)['feature'].tolist()
        important_features = data[top_features].values
        
        # إعداد بيانات التدريب للنموذج المحسن
        train_inputs = {
            'user_id': data['user_id_encoded'].values.reshape(-1, 1),
            'product_id': data['product_id_encoded'].values.reshape(-1, 1),
            'important_features': important_features
        }
        y_train = data['rating'].values
        
        # بناء النموذج المحسن
        num_users = data['user_id_encoded'].nunique()
        num_products = data['product_id_encoded'].nunique()
        model = self.create_feature_optimized_model(num_users, num_products, feature_importance_df)
        
        optimized_model_path = self.model_path.replace('.keras', '_optimized.keras')
        # callbacks
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True, monitor='val_mae', mode='min'),
            ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-6, mode='min'),
            ModelCheckpoint(optimized_model_path, save_best_only=True, monitor='val_mae', mode='min')
        ]
        
        # تدريب النموذج
        history = model.fit(
            train_inputs, y_train,
            batch_size=64,
            epochs=100,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # تحسين أوزان الاندماج
        self.optimize_ensemble_weights(data, model, rf_model)
        
        return model, history

    def optimize_ensemble_weights(self, data, deep_model, rf_model):
        """تحسين أوزان الاندماج بين النماذج"""
        # تحضير البيانات للتنبؤ
        X_rf = data[['age', 'gender', 'location_user', 'category', 'price', 'brand']]
        
        # إعداد مدخلات النموذج العميق
        top_features = ['age', 'price', 'category']  # الميزات الأكثر أهمية
        important_features = data[top_features].values
        
        deep_inputs = {
            'user_id': data['user_id_encoded'].values.reshape(-1, 1),
            'product_id': data['product_id_encoded'].values.reshape(-1, 1),
            'important_features': important_features
        }
        
        # تنبؤات النماذج
        deep_preds = deep_model.predict(deep_inputs).flatten()
        rf_preds = rf_model.predict(X_rf)
        
        # تحسين الأوزان باستخدام التحسين الخطي
        ensemble_inputs = np.column_stack([deep_preds, rf_preds])
        
        # نموذج خطي لإيجاد أفضل الأوزان
        lr = LinearRegression()
        lr.fit(ensemble_inputs, data['rating'].values)
        
        self.ensemble_weights = lr.coef_
        print(f"Optimized ensemble weights: {self.ensemble_weights}")

    def predict_rating(self, raw_input):
        try:
            # معالجة البيانات المدخلة
            processed_input = self.processor.process_input(raw_input)
            
            # ترميز المعرفات
            user_id_encoded = self.user_mapping.get(processed_input['user_id'], 0)
            product_id_encoded = self.product_mapping.get(processed_input['product_id'], 0)
            
            # تحضير مدخلات النموذج العميق
            top_features = ['age', 'price', 'category']
            important_features = np.array([[processed_input.get(f, 0) for f in top_features]], dtype=np.float32)
            
            # إعداد مدخلات النموذج
            inputs = {
                'user_id': np.array([[user_id_encoded]], dtype=np.int32),
                'product_id': np.array([[product_id_encoded]], dtype=np.int32),
                'important_features': important_features
            }

            # التنبؤ بالنموذج العميق
            deep_prediction = self.model.predict(inputs, verbose=0).flatten()[0]
            
            # التنبؤ بـ Random Forest
            rf_input = np.array([[
                processed_input.get('age', 0.5),
                processed_input.get('gender', 0),
                processed_input.get('location_user', 0),
                processed_input.get('category', 0),
                processed_input.get('price', 0.5),
                processed_input.get('brand', 0)
            ]])
            rf_prediction = self.rf_model.predict(rf_input)[0]
            
            # الجمع المرجح
            ensemble_prediction = (self.ensemble_weights[0] * deep_prediction + 
                                  self.ensemble_weights[1] * rf_prediction)
            
            # التأكد من أن التوقع في النطاق الصحيح (1-5)
            final_prediction = np.clip(ensemble_prediction, 1, 5)
            print('##############')
            print(final_prediction)
            print('##############')

            return float(final_prediction)
        
        except Exception as e:
            print(f"فشل في التنبؤ: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def evaluate_model(self, test_size=0.2, cv_folds=5):
            """تقييم النموذج مع التحسينات للنموذج الجديد"""
            try:
                data = pd.read_csv(Config.DATA_PATHS['training_data_merged'])
                
                # تقسيم البيانات مع الحفاظ على توزيع التقييمات
                skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                X = data.drop('rating', axis=1)
                y = data['rating']
                
                rmse_scores, mae_scores, accuracy_scores = [], [], []
                error_analysis = {'Low (1-2)': [], 'Medium (3)': [], 'High (4-5)': []}
                
                for train_index, test_index in skf.split(X, y):
                    train_data, test_data = data.iloc[train_index], data.iloc[test_index]
                    
                    # إعداد بيانات الاختبار باستخدام الدالة المعدلة
                    test_inputs = self._prepare_inputs(test_data)
                    y_test = test_data['rating'].values
                    
                    # التنبؤ
                    predictions = self.model.predict(test_inputs, verbose=0)
                    predicted_classes = np.round(predictions.flatten())  # تقريب للقيم الصحيحة
                    
                    # حساب المقاييس العامة
                    rmse_scores.append(np.sqrt(mean_squared_error(y_test, predicted_classes)))
                    mae_scores.append(mean_absolute_error(y_test, predicted_classes))
                    accuracy_scores.append(accuracy_score(y_test, predicted_classes))
                    
                    # تحليل الأخطاء حسب نطاق التقييم
                    self._analyze_errors_by_range(y_test, predicted_classes, error_analysis)
                
                # حساب المتوسطات
                cv_rmse = np.mean(rmse_scores)
                cv_mae = np.mean(mae_scores)
                cv_accuracy = np.mean(accuracy_scores)
                
                # حساب نسبة التنبؤات ضمن نطاق خطأ معين
                within_margin = np.sum(np.abs(y_test - predicted_classes) <= 1.0)
                accuracy_within_margin = within_margin / len(y_test)
                
                # نتائج تحليل الأخطاء
                error_results = {}
                for range_name, errors in error_analysis.items():
                    if errors:
                        error_results[range_name] = {
                            'Mean Error': np.mean(errors),
                            'Std Error': np.std(errors),
                            'Count': len(errors)
                        }
                
                # تقرير التصنيف
              
                class_report = classification_report(y_test, predicted_classes, output_dict=True, zero_division=0)
                
                # عرض النتائج
                evaluation_results = {
                    'RMSE': cv_rmse,
                    'MAE': cv_mae,
                    'Accuracy': cv_accuracy,
                    'MSE': cv_rmse**2,
                    'Accuracy within ±1': accuracy_within_margin,
                    'Cross Validation RMSE': cv_rmse,
                    'Cross Validation MAE': cv_mae,
                    'Error Analysis': error_results,
                    'Classification Report': class_report
                }
                
                # رسم رسوم بيانية
                self._plot_detailed_evaluation(y_test, predicted_classes, evaluation_results)
                
                return evaluation_results
                    
            except Exception as e:
                print(f"Error in model evaluation: {e}")
                import traceback
                traceback.print_exc()
                raise

    def _analyze_errors_by_range(self, y_true, y_pred, error_analysis):
        """تحليل الأخطاء حسب نطاق التقييم"""
        for actual, predicted in zip(y_true, y_pred):
            error = abs(actual - predicted)
            
            if actual <= 2:
                error_analysis['Low (1-2)'].append(error)
            elif actual == 3:
                error_analysis['Medium (3)'].append(error)
            else:
                error_analysis['High (4-5)'].append(error)

    def _plot_detailed_evaluation(self, y_true, y_pred, results):
        """رسم رسوم بيانية مفصلة للتقييم"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Evaluation', fontsize=16)
        
        # 1. التنبؤات مقابل القيم الفعلية
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
        axes[0, 0].plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
        axes[0, 0].set_xlabel('Actual Ratings')
        axes[0, 0].set_ylabel('Predicted Ratings')
        axes[0, 0].set_title('Predictions vs Actual')
        axes[0, 0].grid(True)
        
        # 2. توزيع الأخطاء
        errors = [abs(a - p) for a, p in zip(y_true, y_pred)]
        axes[0, 1].hist(errors, bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 1].axvline(np.mean(errors), color='red', linestyle='dashed', linewidth=2)
        axes[0, 1].set_xlabel('Absolute Error')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Error Distribution')
        axes[0, 1].grid(True)
        
        # 3. مقاييس الأداء
        metrics = ['RMSE', 'MAE', 'Accuracy']
        values = [results[m] for m in metrics]
        bars = axes[1, 0].bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral'])
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].set_title('Performance Metrics')
        for i, v in enumerate(values):
            axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
        
        # 4. الأخطاء حسب نطاق التقييم
        error_data = results['Error Analysis']
        ranges = list(error_data.keys())
        mean_errors = [error_data[r]['Mean Error'] for r in ranges]
        
        bars = axes[1, 1].bar(ranges, mean_errors, color=['lightblue', 'lightgreen', 'salmon'])
        axes[1, 1].set_ylabel('Mean Absolute Error')
        axes[1, 1].set_title('Error by Rating Range')
        for i, v in enumerate(mean_errors):
            axes[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 5. مصفوفة الارتباك
        self._plot_confusion_matrix(y_true, y_pred)
        
        # طباعة التقرير
        print("=" * 60)
        print("DEEP LEARNING MODEL EVALUATION REPORT")
        print("=" * 60)
        print(f"RMSE: {results['RMSE']:.4f}")
        print(f"MAE: {results['MAE']:.4f}")
        print(f"Accuracy: {results['Accuracy']:.4f}")
        print(f"MSE: {results['MSE']:.4f}")
        print(f"Accuracy within ±1: {results['Accuracy within ±1']:.2%}")
        print(f"Cross Validation RMSE: {results['Cross Validation RMSE']:.4f}")
        print(f"Cross Validation MAE: {results['Cross Validation MAE']:.4f}")
        
        print("\nError Analysis by Rating Range:")
        for range_name, stats in results['Error Analysis'].items():
            print(f"  {range_name}:")
            print(f"    Mean Error: {stats['Mean Error']:.3f}")
            print(f"    Std Error: {stats['Std Error']:.3f}")
            print(f"    Sample Count: {stats['Count']}")
            print()
            
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))

    def _plot_confusion_matrix(self, y_true, y_pred):
        """رسم مصفوفة الارتباك"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=[1, 2, 3, 4, 5], 
                   yticklabels=[1, 2, 3, 4, 5])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _prepare_inputs(self, data):
        """إعداد مدخلات النموذج من البيانات - نسخة معدلة للنموذج الجديد"""
        data1 = data.copy()
        # ترميز المعرفات
        if 'user_id' in data.columns and hasattr(self, 'user_mapping'):
            data1.loc[:, 'user_id_encoded'] = data1['user_id'].map(self.user_mapping).fillna(0)
        
        if 'product_id' in data.columns and hasattr(self, 'product_mapping'):
            data1.loc[:, 'product_id_encoded'] = data1['product_id'].map(self.product_mapping).fillna(0)
        
        # تحديد الميزات المهمة بناءً على تحليل Random Forest
        important_features = ['age', 'price', 'category']  
        
        for feature in important_features:
            if feature not in data1.columns:
                data1.loc[:, feature] = 0
        # إعداد المدخلات للنموذج الجديد
        inputs = {
            'user_id': data1['user_id_encoded'].values.reshape(-1, 1).astype(np.int32),
            'product_id': data1['product_id_encoded'].values.reshape(-1, 1).astype(np.int32),
            'important_features': data1[important_features].values.astype(np.float32)
        }
        
        return inputs
    
    def try_alternative_approach(self):
        # تجريب نموذج بسيط أولاً للتحقق من جودة البيانات
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        
        data = self.preprocess_data()
        
        # استخدام الميزات فقط (بدون embedding)
        X = data[['age', 'gender', 'location_user', 'category', 'price', 'brand']]
        y = data['rating']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # نموذج Random Forest بسيط
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # تقييم الأداء
        predictions = rf_model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        
        print(f"Random Forest MAE: {mae}")
        print(f"Random Forest RMSE: {rmse}")
        
        # إذا كان Random Forest يعمل بشكل أفضل، فقد تكون المشكلة في بنية النموذج العميق
        return rf_model

# باقي الكود (الطبقات المخصصة ووظيفة الموازنة)
class SafeCastLayer(tf.keras.layers.Layer):
    def __init__(self, output_dtype='int32', **kwargs):
        super(SafeCastLayer, self).__init__(**kwargs)
        self.output_dtype = output_dtype
        
    def call(self, inputs):
        return tf.cast(inputs, self.output_dtype)
    
    def get_config(self):
        config = super(SafeCastLayer, self).get_config()
        config.update({'output_dtype': self.output_dtype})
        return config

class SafeClipLayer(tf.keras.layers.Layer):
    def __init__(self, min_value, max_value, **kwargs):
        super(SafeClipLayer, self).__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value
        
    def call(self, inputs):
        return tf.clip_by_value(inputs, self.min_value, self.max_value)
    
    def get_config(self):
        config = super(SafeClipLayer, self).get_config()
        config.update({
            'min_value': self.min_value,
            'max_value': self.max_value
        })
        return config


