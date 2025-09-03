
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
import logging

class DataProcessor:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.default_values = {}  # لتخزين القيم الافتراضية
        self.logger = logging.getLogger(__name__)
        self.user_mapping = None
        self.product_mapping = None

    def fit(self, data):
        """تدريب معالجات البيانات مع تسجيل القيم الافتراضية"""
        # حفظ مضايفات المستخدمين والمنتجات
        if 'user_id' in data.columns:
            self.user_mapping = {user_id: idx for idx, user_id in enumerate(data['user_id'].unique())}
        
        if 'product_id' in data.columns:
            self.product_mapping = {product_id: idx for idx, product_id in enumerate(data['product_id'].unique())}

        # المعالجة الرقمية
        num_cols = ['age', 'price']
        for col in num_cols:
            if col in data.columns:
                self.scalers[col] = StandardScaler()
                self.scalers[col].fit(data[[col]])
                self.default_values[col] = data[col].median()  # تخزين الوسيط كقيمة افتراضية

        # المعالجة الفئوية
        cat_cols = ['gender', 'location_user', 'category', 'brand']
        for col in cat_cols:
            if col in data.columns:
                self.encoders[col] = LabelEncoder()
                self.encoders[col].fit(data[col].astype(str))
                # تخزين القيمة الأكثر تكرارا كافتراضية
                self.default_values[col] = data[col].mode()[0] if not data[col].mode().empty else 'unknown'

    def safe_transform(self, encoder, values, col_name):
        """تحويل آمن للبيانات الفئوية مع التعامل مع القيم الجديدة"""
        try:
            return encoder.transform(values)
        except ValueError:
            # إذا كانت القيمة جديدة، نستخدم قيمة افتراضية
            default_val = self.default_values.get(col_name, 'Unknown')
            return encoder.transform([default_val] * len(values))

    def process_input(self, raw_input):
        """إصدار معدل مع تحسينات تتبع الأخطاء"""
        try:
            # 1. التحقق من وجود الحقول الأساسية
            required_fields = ['user_id', 'product_id', 'age', 'gender', 'location_user', 'category', 'price', 'brand']
            for field in required_fields:
                if field not in raw_input:
                    self.logger.error(f"الحقل المطلوب {field} مفقود")
                    raise ValueError(f"الحقل المطلوب {field} مفقود")

            # 2. إنشاء قاموس المعالجة
            processed = {
                'user_id': self.user_mapping.get(int(raw_input['user_id']), 0) if self.user_mapping else 0,
                'product_id': self.product_mapping.get(int(raw_input['product_id']), 0) if self.product_mapping else 0
            }

            # 3. المعالجة الرقمية مع التحقق
            if 'age' in self.scalers:
                processed['age'] = self.scalers['age'].transform([[float(raw_input['age'])]])[0][0]
            else:
                raise ValueError("Scaler للعمر غير موجود")

            if 'price' in self.scalers:
                processed['price'] = self.scalers['price'].transform([[float(raw_input['price'])]])[0][0]
            else:
                raise ValueError("Scaler للسعر غير موجود")

            # 4. المعالجة الفئوية مع التحقق
            for col in ['gender', 'location_user', 'category', 'brand']:
                if col in self.encoders:
                    processed[col] = self.encoders[col].transform([str(raw_input[col])])[0]
                else:
                    processed[col] = 0  # قيمة افتراضية

            return processed

        except Exception as e:
            self.logger.error(f"خطأ في معالجة المدخلات: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def process_batch(self, batch_data):
        """معالجة دفعة كاملة مع التعامل مع القيم الجديدة"""
        processed = batch_data.copy()

        # ترميز المعرفات
        if 'user_id' in processed.columns and self.user_mapping:
            processed['user_id'] = processed['user_id'].map(self.user_mapping).fillna(0)
        
        if 'product_id' in processed.columns and self.product_mapping:
            processed['product_id'] = processed['product_id'].map(self.product_mapping).fillna(0)

        # المعالجة الرقمية
        for col in ['age', 'price']:
            if col in self.scalers and col in processed.columns:
                # استبدال القيم المفقودة بالقيمة الافتراضية
                processed[col] = processed[col].fillna(self.default_values[col])
                processed[col] = self.scalers[col].transform(processed[[col]])

        # المعالجة الفئوية
        for col, encoder in self.encoders.items():
            if col in processed.columns:
                # تحويل آمن مع التعامل مع القيم الجديدة
                processed[col] = processed[col].astype(str).fillna(str(self.default_values[col]))
                processed[col] = self.safe_transform(encoder, processed[col], col)

        return processed

    def handle_missing_values(self, raw_input):
        """معالجة القيم المفقودة في البيانات"""
        defaults = {
            'user_id': 0,
            'product_id': 0,
            'age': 30,
            'gender': 'unknown',
            'location_user': 'unknown',
            'category': 'other',
            'price': 0,
            'brand': 'unknown'
        }

        for key in defaults:
            if key not in raw_input or raw_input[key] is None:
                raw_input[key] = defaults[key]

        return raw_input

    def validate_input(self, raw_input):
        """التحقق من صحة البيانات المدخلة"""
        required = {
            'user_id': int,
            'product_id': int,
            'age': (int, float),
            'gender': str,
            'location_user': str,
            'category': str,
            'price': (int, float),
            'brand': str
        }

        for field, types in required.items():
            if field not in raw_input:
                return False, f"الحقل {field} مفقود"
            if not isinstance(raw_input[field], types):
                return False, f"الحقل {field} يجب أن يكون من نوع {types}"

        return True, "البيانات صالحة"

    def get_default_values(self):
        """الحصول على القيم الافتراضية للاستخدام عند وجود أخطاء"""
        defaults = {
            'user_id': 0,
            'product_id': 0,
            'age': 30,
            'gender': 'unknown',
            'location_user': 'unknown',
            'category': 'other',
            'price': 0,
            'brand': 'unknown'
        }
        
        # تحديث بالقيم التي تم تعلمها من البيانات
        for key in self.default_values:
            if key in defaults:
                defaults[key] = self.default_values[key]
                
        return defaults

    def check_processor(self):
        """فحص حالة المعالج قبل الاستخدام"""
        print("\n=== فحص المعالج ===")

        # 1. التحقق من السكالرز
        if not hasattr(self, 'scalers'):
            print("تحذير: لا يوجد scalers في المعالج")
        else:
            print("Scalers المتاحة:", list(self.scalers.keys()))

        # 2. التحقق من الانكودرز
        if not hasattr(self, 'encoders'):
            print("تحذير: لا يوجد encoders في المعالج")
        else:
            print("Encoders المتاحة:", list(self.encoders.keys()))

        # 3. التحقق من القيم الافتراضية
        if not hasattr(self, 'default_values'):
            print("تحذير: لا يوجد default_values في المعالج")
        else:
            print("بعض القيم الافتراضية:", {k: self.default_values[k] for k in list(self.default_values.keys())[:3]})
            
        # 4. التحقق من  المستخدمين والمنتجات
        print("User mapping:", "موجود" if self.user_mapping else "غير موجود")
        print("Product mapping:", "موجود" if self.product_mapping else "غير موجود")