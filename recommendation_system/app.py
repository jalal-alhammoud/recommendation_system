from flask import Flask, request, jsonify, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from flask_bcrypt import Bcrypt
from flask_caching import Cache
from flask_compress import Compress
from configdb.configdb import Configdb
from database.models import db, bcrypt, User, Category, Product, Review, Cart, CartItem
import os
from werkzeug.utils import secure_filename
import uuid
from flask_cors import CORS
from flask_jwt_extended import jwt_required, get_jwt_identity
from werkzeug.exceptions import HTTPException, NotFound
from datetime import datetime as dt
import pandas as pd
from recommendation_system.functions.recommendingFuctions import load_and_preprocess_data
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from recommendation_system.functions.recommendingFuctions import *
from recommendation_system.modelServices.svdModel import SvdModel
from recommendation_system.modelServices.knnModel import KnnModel
from recommendation_system.modelServices.deepLearnModel import DeepLearnModel
from recommendation_system.imageRecommender.imageRecommender import ImageRecommendationService

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')


users = pd.read_csv(os.path.join(DATA_DIR, 'users.csv'))
context = pd.read_csv(os.path.join(DATA_DIR, 'context.csv'))
images_data = pd.read_csv(os.path.join(DATA_DIR, 'images_data.csv'))
interactions = pd.read_csv(os.path.join(DATA_DIR, 'interactions.csv'))
products = pd.read_csv(os.path.join(DATA_DIR, 'products.csv'))
reviews = pd.read_csv(os.path.join(DATA_DIR, 'reviews.csv'))
metadata = load_and_preprocess_data()


service = ImageRecommendationService()


# Initialize Flask app
app = Flask(__name__)
CORS(app,supports_credentials=True)  
CORS(app, 
     resources={
         r"/api/*": {
             "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
             "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
             "allow_headers": ["Content-Type", "Authorization"],
             "supports_credentials": True
         }
     })
app.config.from_object(Configdb)


# Initialize extensions
db.init_app(app)

cache = Cache(config={'CACHE_TYPE': 'SimpleCache'})
cache.init_app(app)
compress = Compress()
compress.init_app(app)

app.config.update(
    SQLALCHEMY_ENGINE_OPTIONS={
        "pool_pre_ping": True,
        "pool_recycle": 300,
        "pool_size": 20,
        "max_overflow": 30,
    },
    JSONIFY_PRETTYPRINT_REGULAR=False,  
    SQLALCHEMY_TRACK_MODIFICATIONS=False
)
ma = Marshmallow(app)
bcrypt.init_app(app)

# Create database tables
with app.app_context():
    db.create_all()

engine = create_engine(app.config['SQLALCHEMY_DATABASE_URI'])
db_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
# إعدادات رفع الملفات
UPLOAD_FOLDER = 'static/uploads/users'
CATEGORY_UPLOAD_FOLDER = 'static/uploads/categories'
PRODUCT_UPLOAD_FOLDER = 'static/uploads/products'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CATEGORY_UPLOAD_FOLDER'] = CATEGORY_UPLOAD_FOLDER
app.config['PRODUCT_UPLOAD_FOLDER'] = PRODUCT_UPLOAD_FOLDER
# تأكد من وجود مجلد التحميل
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['CATEGORY_UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PRODUCT_UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Schemas
class UserSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = User
        exclude = ('password',)

class CategorySchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = Category

class ProductSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = Product
        include_fk = True
    
    category = ma.Nested(CategorySchema)

class ReviewSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = Review
        include_fk = True
    
    user = ma.Nested(UserSchema)
    product = ma.Nested(ProductSchema)

class CartItemSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = CartItem
        include_fk = True
    
    product = ma.Nested(ProductSchema)

class CartSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = Cart
    
    items = ma.Nested(CartItemSchema, many=True)

# Initialize schemas
user_schema = UserSchema()
users_schema = UserSchema(many=True)
category_schema = CategorySchema()
categories_schema = CategorySchema(many=True)
product_schema = ProductSchema()
products_schema = ProductSchema(many=True)
review_schema = ReviewSchema()
reviews_schema = ReviewSchema(many=True)
cart_schema = CartSchema()
cart_item_schema = CartItemSchema()

_first_request_handled = False

# Auth Routes
@app.route('/api/signup', methods=['POST'])
def signup():
    # التحقق من وجود بيانات JSON وملف الصورة
    if 'user_image' in request.files:
        file = request.files['user_image']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # إنشاء اسم فريد للملف
            unique_filename = f"{str(uuid.uuid4())}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            user_image = unique_filename
        else:
            return jsonify({'message': 'Invalid file type. Only images are allowed'}), 400
    else:
        user_image = None

    # الحصول على باقي البيانات من form-data
    data = request.form
    
    # التحقق من وجود البيانات المطلوبة
    if not all(k in data for k in ['username', 'password', 'email']):
        return jsonify({'message': 'Missing required fields'}), 400
    
    # التحقق من عدم تكرار اسم المستخدم أو البريد الإلكتروني
    if User.query.filter_by(user_name=data['username']).first():
        return jsonify({'message': 'Username already exists'}), 400
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'message': 'Email already exists'}), 400
    
    # إنشاء مستخدم جديد
    new_user = User(
        user_name=data['username'],
        password=data['password'],
        email=data['email'],
        age=data.get('age'),
        gender=data.get('gender'),
        location=data.get('location'),
        user_image=user_image,  # اسم الملف الذي تم حفظه
        admin=data.get('admin', False)
    )
    
    db.session.add(new_user)
    db.session.commit()
    
    #  add to data
    new_data = {
        'user_id': new_user.user_id,
        'age': data.get('age'),
        'gender': data.get('gender'),
        'location': data.get('location')
    }
    global users
    users = pd.concat([users, pd.DataFrame([new_data])], ignore_index=True)
    users.to_csv(os.path.join(DATA_DIR, 'users.csv'), index=False) 

    return jsonify({
        'message': 'User created successfully',
        'user_id': new_user.user_id,
        'user_image': user_image
    }), 201

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    identifier = data.get('identifier')  # Can be username or email
    password = data.get('password')
    
    user = User.query.filter((User.user_name == identifier) | (User.email == identifier)).first()
    
    if user and user.check_password(password):
       
        return jsonify({
            'message': 'Login successful',
            'user': user_schema.dump(user)
        }), 200
    
    return jsonify({'message': 'Invalid credentials'}), 401

@app.route('/api/user/<int:user_id>', methods=['GET'])
def get_user_profile(user_id):
    try:
        user = User.query.get_or_404(user_id)
        return jsonify({
            "user": {
                "user_id": user.user_id,
                "username": user.user_name,
                "email": user.email,
                "age": user.age,
                "gender": user.gender,
                "location": user.location,
                "user_image": url_for('static', filename=f'uploads/users/{user.user_image}', _external=True) if user.user_image else None,
                "admin": user.admin
            }
        }), 200
        
    except Exception as e:
        app.logger.error(f"Error in user profile: {str(e)}")
        return jsonify({"msg": "Server error"}), 500

@app.route('/api/user/updateprofile', methods=['POST'])
def update_user():
    data = request.form 
    user_id = data['user_id']
    user = User.query.get_or_404(user_id)
    
    # معالجة ملف الصورة إذا تم رفعه
    if 'user_image' in request.files:
        file = request.files['user_image']
        if file and allowed_file(file.filename):
            # حذف الصورة القديمة إذا كانت موجودة
            if user.user_image:
                try:
                    old_file_path = os.path.join(app.config['UPLOAD_FOLDER'], user.user_image)
                    if os.path.exists(old_file_path):
                        os.remove(old_file_path)
                except Exception as e:
                    app.logger.error(f"Error deleting old image: {e}")
            
            # حفظ الصورة الجديدة
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4().hex}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            user.user_image = unique_filename
    
    # التحقق من تحديث اسم المستخدم
    if 'username' in data and data['username'] != user.user_name:
        if User.query.filter_by(user_name=data['username']).first():
            return jsonify({'message': 'Username already exists'}), 400
        user.user_name = data['username']
    
    # التحقق من تحديث البريد الإلكتروني
    if 'email' in data and data['email'] != user.email:
        if User.query.filter_by(email=data['email']).first():
            return jsonify({'message': 'Email already exists'}), 400
        user.email = data['email']
    
    # تحديث كلمة المرور إذا تم توفيرها
    if 'password' in data:
        user.set_password(data['password'])
    
    # تحديث البيانات الأخرى
    if 'age' in data:
        user.age = int(data['age']) if data['age'] else None
    
    if 'gender' in data:
        user.gender = data['gender']
    
    if 'location' in data:
        user.location = data['location']
    
    # تحديث صلاحية الأدمن (فقط بواسطة أدمن آخر)
    if 'admin' in data :
        user.admin = data['admin'].lower() == 'true'
    
    db.session.commit()
    
    #add users to data
    global users
    user_index = users[users['user_id'] == int(user_id)].index
    
    if not user_index.empty:
        if 'age' in data:
            users.at[user_index[0], 'age'] = int(data['age']) if data['age'] else None
        if 'gender' in data:
            users.at[user_index[0], 'gender'] = data['gender']
        if 'location' in data:
            users.at[user_index[0], 'location'] = data['location']
        
        # حفظ التغييرات في الملف (اختياري)
        users.to_csv(os.path.join(DATA_DIR, 'users.csv'), index=False)


    return jsonify({
        'message': 'User updated successfully',
        'user': user_schema.dump(user)
    })


# Category Routes
@app.route('/api/categories', methods=['POST'])
def add_category():
    
    # معالجة ملف الصورة إذا تم رفعه
    cat_image = None
    if 'Cat_image' in request.files:
        file = request.files['Cat_image']
        if file and allowed_file(file.filename):
            # إنشاء اسم فريد للملف
            filename = secure_filename(file.filename)
            unique_filename = f"cat_{uuid.uuid4().hex}_{filename}"
            file_path = os.path.join(app.config['CATEGORY_UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            cat_image = unique_filename
        else:
            return jsonify({'message': 'Invalid file type. Allowed extensions are: png, jpg, jpeg, gif'}), 400
    
    # الحصول على باقي البيانات
    data = request.form if 'Cat_image' in request.files else request.get_json()
    
    # التحقق من البيانات المطلوبة
    if 'Cat_name' not in data:
        return jsonify({'message': 'Category name is required'}), 400
    
    # إنشاء الفئة الجديدة
    new_category = Category(
        cat_name=data['Cat_name'],
        description=data.get('Description'),
        cat_image=cat_image
    )
    
    db.session.add(new_category)
    db.session.commit()
    
    return jsonify({
        'message': 'Category added successfully',
        'category': category_schema.dump(new_category)
    }), 201

@app.route('/api/categories/update_category', methods=['POST'])
def update_category():
    data = request.form 
    cat_id = data['Cat_id'] 
    category = Category.query.get_or_404(data['Cat_id'])
    
    # معالجة ملف الصورة إذا تم رفعه
    if 'Cat_image' in request.files:
        file = request.files['Cat_image']
        if file and allowed_file(file.filename):
            # حذف الصورة القديمة إذا كانت موجودة
            if category.cat_image:
                try:
                    old_image_path = os.path.join(app.config['CATEGORY_UPLOAD_FOLDER'], category.cat_image)
                    if os.path.exists(old_image_path):
                        os.remove(old_image_path)
                except Exception as e:
                    app.logger.error(f"Error deleting old category image: {e}")
            
            # حفظ الصورة الجديدة
            filename = secure_filename(file.filename)
            unique_filename = f"cat_{uuid.uuid4().hex}_{filename}"
            file_path = os.path.join(app.config['CATEGORY_UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            category.cat_image = unique_filename
    
    if 'Cat_name' in data and data['Cat_name'] != category.cat_name:
        # التحقق من عدم تكرار اسم الفئة
        existing_category = Category.query.filter_by(cat_name=data['Cat_name']).first()
        if existing_category and existing_category.cat_id != cat_id:
            return jsonify({'message': 'Category name already exists'}), 400
        category.cat_name = data['Cat_name']
    
    if 'Description' in data:
        category.description = data['Description']
    
    db.session.commit()
    
    return jsonify({
        'message': 'Category updated successfully',
        'category': category_schema.dump(category)
    })
@app.route('/api/categories/<int:cat_id>', methods=['DELETE'])
def delete_category(cat_id):
    
    category = Category.query.get_or_404(cat_id)
    
    # التحقق من وجود منتجات مرتبطة بهذه الفئة
    if category.products:
        return jsonify({
            'message': 'Cannot delete category with associated products',
            'products_count': len(category.products)
        }), 400
    
    # حذف صورة الفئة إذا كانت موجودة
    if category.Cat_image:
        try:
            image_path = os.path.join(app.config['CATEGORY_UPLOAD_FOLDER'], category.cat_image)
            if os.path.exists(image_path):
                os.remove(image_path)
                        
        except OSError as e:
            app.logger.error(f"Error deleting category image: {e}")
            return jsonify({
                'message': 'Category deleted but image cleanup failed',
                'error': str(e)
            }), 500
    
    # حذف الفئة من قاعدة البيانات
    db.session.delete(category)
    db.session.commit()
    
    return jsonify({
        'message': 'Category deleted successfully',
        'deleted_category_id': cat_id
    })

@app.route('/api/categories/<int:cat_id>', methods=['GET'])
def get_category(cat_id):
    category = Category.query.get_or_404(cat_id)
    return jsonify({'category': category_schema.dump(category)})

@app.route('/api/categories', methods=['GET'])
@cache.cached(timeout=300, query_string=True)
def get_all_categories():
    categories = Category.query.all()
    return jsonify({'categories': categories_schema.dump(categories)})

# Product Routes
@app.route('/api/products', methods=['POST'])
def add_product():
    # معالجة ملف الصورة إذا تم رفعه
    product_image = None
    if 'product_image' in request.files:
        file = request.files['product_image']
        if file and allowed_file(file.filename):
            # إنشاء اسم فريد للملف
            filename = secure_filename(file.filename)
            unique_filename = f"prod_{uuid.uuid4().hex}_{filename}"
            file_path = os.path.join(app.config['PRODUCT_UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            product_image = unique_filename
        else:
            return jsonify({
                'message': 'Invalid file type. Allowed extensions are: png, jpg, jpeg, gif'
            }), 400
    
    # الحصول على باقي البيانات
    data = request.form if 'product_image' in request.files else request.get_json()
    
    # التحقق من البيانات المطلوبة
    if not all(k in data for k in ['product_name', 'price']):
        return jsonify({'message': 'Product name and price are required'}), 400
    
    # التحقق من وجود الفئة إذا تم تحديدها
    if 'cat_id' in data:
        category = Category.query.get(data['cat_id'])
        if not category:
            return jsonify({'message': 'Category not found'}), 404
        category_name = category.cat_name
    # إنشاء المنتج الجديد
    new_product = Product(
        product_name=data['product_name'],
        description=data.get('description'),
        price=data['price'],
        country_made=data.get('country_made'),
        product_image=product_image,  # اسم الملف الذي تم حفظه
        brand=data.get('brand'),
        cat_id=data.get('cat_id')
    )
    
    db.session.add(new_product)
    db.session.commit()
    
    # إضافة المنتج إلى DataFrame products
    new_product_data = {
        'product_id': new_product.product_id,
        'product_name': data['product_name'],
        'price': float(data['price']),
        'category': category_name,
        'country_made': data.get('country_made', 'Unknown'),
        'brand': data.get('brand', 'Unknown')
    }
    global products
    global images_data
    products = pd.concat([products, pd.DataFrame([new_product_data])], ignore_index=True)

    # إضافة الصورة إلى DataFrame images_data إذا وجدت
    if product_image:
        new_image_data = {
            'product_id': new_product.product_id,
            'images': product_image
        }      
       
        images_data = pd.concat([images_data, pd.DataFrame([new_image_data])], ignore_index=True)
    products.to_csv(os.path.join(DATA_DIR, 'products.csv'), index=False)
    if product_image:
        images_data.to_csv(os.path.join(DATA_DIR, 'images_data.csv'), index=False)

    return jsonify({
        'message': 'Product added successfully',
        'product': product_schema.dump(new_product)
    }), 201

@app.route('/api/products/update_product', methods=['POST'])
def update_product():
    data = request.form
    product = Product.query.get_or_404(data['product_id'])
    try:
        image_updated = False
        # معالجة ملف الصورة إذا تم رفعه
        if 'product_image' in request.files:
            file = request.files['product_image']
            if file and allowed_file(file.filename):
                # حذف الصورة القديمة إذا كانت موجودة
                if product.product_image:
                    try:
                        old_image_path = os.path.join(app.config['PRODUCT_UPLOAD_FOLDER'], product.product_image)
                        if os.path.exists(old_image_path):
                            os.remove(old_image_path)
                    except Exception as e:
                        app.logger.error(f"Error deleting old product image: {e}")
                
                # حفظ الصورة الجديدة
                filename = secure_filename(file.filename)
                unique_filename = f"prod_{uuid.uuid4().hex}_{filename}"
                file_path = os.path.join(app.config['PRODUCT_UPLOAD_FOLDER'], unique_filename)
                file.save(file_path)
                product.product_image = unique_filename
                image_updated = True
            elif file:  # إذا كان الملف موجوداً ولكن غير مسموح بنوعه
                return jsonify({'message': 'Invalid file type. Only images are allowed'}), 400
   
        # تحديث بيانات المنتج
        if 'product_name' in data:
            product.product_name = data['product_name']
        
        if 'description' in data:
            product.description = data['description']
        
        if 'price' in data:
            try:
                product.price = float(data['price'])
            except ValueError:
                return jsonify({'message': 'Invalid price format'}), 400
        
        if 'country_made' in data:
            product.country_made = data['country_made']
        
        if 'brand' in data:
            product.brand = data['brand']
        
        if 'cat_id' in data:
            category = Category.query.get(data['cat_id'])
            if not category:
                return jsonify({'message': 'Category not found'}), 404
            product.cat_id = data['cat_id']
            category_name = category.cat_name
        
        db.session.commit()
        
        global products
        global images_data
        product_index = products[products['product_id'] == product.product_id].index
        
        if not product_index.empty:
            # تحديث البيانات الأساسية
            if 'product_name' in data:
                products.at[product_index[0], 'product_name'] = data['product_name']
            
            if 'price' in data:
                products.at[product_index[0], 'price'] = float(data['price'])
            
            if 'country_made' in data:
                products.at[product_index[0], 'country_made'] = data['country_made']
            
            if 'brand' in data:
                products.at[product_index[0], 'brand'] = data['brand']
            
            # تحديث اسم الفئة إذا تغيرت
            if category_name:
                products.at[product_index[0], 'category'] = category_name
            
            # حفظ التغييرات
            products.to_csv(os.path.join(DATA_DIR, 'products.csv'), index=False)
        
        # تحديث DataFrame images_data إذا تغيرت الصورة
        if image_updated:
            image_index = images_data[images_data['product_id'] == product.product_id].index
            
            if not image_index.empty:
                # تحديث الصورة الحالية
                images_data.at[image_index[0], 'images'] = product.product_image
            else:
                # أو إضافة صورة جديدة إذا لم تكن موجودة
                new_image_data = {
                    'product_id': product.product_id,
                    'images': product.product_image
                }
                images_data = pd.concat([images_data, pd.DataFrame([new_image_data])], ignore_index=True)
            
            images_data.to_csv(os.path.join(DATA_DIR, 'images_data.csv'), index=False)


        return jsonify({
                'message': 'Product updated successfully',
                'product': product_schema.dump(product)
            })
    
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error updating product: {str(e)}")
        return jsonify({'message': 'Internal server error', 'error': str(e)}), 500
    

@app.route('/api/products/<int:product_id>', methods=['DELETE'])
def delete_product(product_id):
   
    product = Product.query.get_or_404(product_id)
    
    # حذف صورة المنتج إذا كانت موجودة
    if product.product_image:
        try:
            image_path = os.path.join(app.config['PRODUCT_UPLOAD_FOLDER'], product.product_image)
            if os.path.exists(image_path):
                os.remove(image_path)
                
                        
        except OSError as e:
            app.logger.error(f"Error deleting product image: {e}")
            return jsonify({
                'message': 'Product deleted but image cleanup failed',
                'error': str(e)
            }), 500
    
    # حذف أي تقييمات مرتبطة بالمنتج 
    Review.query.filter_by(product_id=product_id).delete()
    
    global products
    global reviews
    global context
    global images_data
    global interactions

    
    # حذف من DataFrame products
    products = products[products['product_id'] != product_id]

    # حذف المنتج من reviews
    reviews = reviews[reviews['product_id'] != product_id]

    # حذف المنتج من contextو interactions
    index_to_delete = interactions[interactions['product_id'] == product_id].index
   
    interactions = interactions.drop(index_to_delete)
    context = context.drop(index_to_delete)
    interactions = interactions.reset_index(drop=True)
    context = context.reset_index(drop=True)
    interactions.to_csv(os.path.join(DATA_DIR, 'interactions.csv'), index=False)
    context.to_csv(os.path.join(DATA_DIR, 'context.csv'), index=False)
        
    
    # حذف من DataFrame images_data
    images_data = images_data[images_data['product_id'] != product_id]
        
    # حفظ التغييرات في ملفات CSV
    products.to_csv(os.path.join(DATA_DIR, 'products.csv'), index=False)
    images_data.to_csv(os.path.join(DATA_DIR, 'images_data.csv'), index=False)
    reviews.to_csv(os.path.join(DATA_DIR, 'reviews.csv'), index=False)

    # حذف المنتج من قاعدة البيانات
    db.session.delete(product)
    db.session.commit()
    
    return jsonify({
        'message': 'Product deleted successfully',
        'deleted_product_id': product_id
    })

@app.route('/api/products/<int:product_id>', methods=['GET'])
def get_product(product_id):
    product = Product.query.get_or_404(product_id)
    return jsonify({'product': product_schema.dump(product)})


@app.route('/api/products', methods=['GET'])
@cache.cached(timeout=400, query_string=True)
def get_products():
    limit = request.args.get('limit', default=None, type=int)
    if limit:
        products = Product.query.limit(limit).all()
    else:
        products = Product.query.all()
    return jsonify({'products': products_schema.dump(products)})

@app.route('/api/products/category/<int:cat_id>', methods=['GET'])
def get_products_by_category(cat_id):
    products = Product.query.filter_by(cat_id=cat_id).all()
    return jsonify({'products': products_schema.dump(products)})

@app.route('/api/products/country/<string:country>', methods=['GET'])
def get_products_by_country(country):
    products = Product.query.filter_by(country_made=country).all()
    return jsonify({'products': products_schema.dump(products)})

@app.route('/api/products/brand/<string:brand>', methods=['GET'])
def get_products_by_brand(brand):
    products = Product.query.filter_by(brand=brand).all()
    return jsonify({'products': products_schema.dump(products)})

@app.route('/api/brands', methods=['GET'])
@cache.cached(timeout=400, query_string=True)
def get_all_brands():
    brands = db.session.query(Product.brand).distinct().all()
    brands = [brand[0] for brand in brands if brand[0]]
    return jsonify({'brands': brands})

@app.route('/api/products/search', methods=['GET'])
def search_products():
    query = request.args.get('q')
    if not query:
        return jsonify({'message': 'No search query provided'}), 400
    
    products = Product.query.filter(
        Product.product_name.ilike(f'%{query}%') | 
        Product.description.ilike(f'%{query}%')
    ).all()
    
    return jsonify({'products': products_schema.dump(products)})

# Review Routes
@app.route('/api/reviews', methods=['POST'])
def add_review():
    data = request.get_json()
    
    # التحقق من وجود المنتج
    product = Product.query.get(data['product_id'])
    if not product:
        return jsonify({'message': 'Product not found'}), 404
    
    # البحث عن التقييم الحالي إذا كان موجوداً
    existing_review = Review.query.filter_by(
        user_id=data['user_id'],
        product_id=data['product_id']
    ).first()

    global reviews
    if existing_review:
        # تحديث التقييم الحالي بدلاً من إنشاء جديد
        if 'rating' in data:
            existing_review.rating = data['rating']
        if 'review_text' in data:
            existing_review.review_text = data['review_text']
        
        db.session.commit()
        # تحديث التقييم في DataFrame reviews
      
        review_index = reviews[
            (reviews['user_id'] == data['user_id']) & 
            (reviews['product_id'] == data['product_id'])
        ].index

        if not review_index.empty:
            if 'rating' in data:
                reviews.at[review_index[0], 'rating'] = data['rating']
            if 'review_text' in data:
                reviews.at[review_index[0], 'review_text'] = data['review_text']
            reviews.to_csv(os.path.join(DATA_DIR, 'reviews.csv'), index=False)

    
        return jsonify({
            'message': 'Review updated successfully',
            'review': review_schema.dump(existing_review)
        }), 200
    else:
        # إنشاء تقييم جديد إذا لم يكن موجوداً
        new_review = Review(
            user_id=data['user_id'],
            product_id=data['product_id'],
            rating=data['rating'],
            review_text=data.get('review_text')
        )
        
        db.session.add(new_review)
        db.session.commit()

         # إضافة التقييم الجديد إلى DataFrame reviews
        new_review_data = {
            'user_id': data['user_id'],
            'product_id': data['product_id'],
            'rating': data['rating'],
            'review_text': data.get('review_text')
        }
        
       
        reviews = pd.concat([reviews, pd.DataFrame([new_review_data])], ignore_index=True)
        reviews.to_csv(os.path.join(DATA_DIR, 'reviews.csv'), index=False)
        return jsonify({
            'message': 'Review added successfully',
            'review': review_schema.dump(new_review)
        }), 201

@app.route('/api/reviews/user/<int:user_id>', methods=['GET'])
def get_user_reviews(user_id):
    reviews = Review.query.filter_by(user_id=user_id).all()
    return jsonify({'reviews': reviews_schema.dump(reviews)})

@app.route('/api/reviews/product/<int:product_id>', methods=['GET'])
def get_product_reviews(product_id):
    reviews = Review.query.filter_by(product_id=product_id).all()
    return jsonify({'reviews': reviews_schema.dump(reviews)})

@app.route('/api/reviews/user/<int:user_id>/product/<int:product_id>', methods=['GET'])
def get_user_product_review(user_id, product_id):
    review = Review.query.filter_by(user_id=user_id, product_id=product_id).first()
    if not review:
        return jsonify({'message': 'Review not found'}), 404
    return jsonify({'review': review_schema.dump(review)})

# Cart Routes
@app.route('/api/cart', methods=['POST'])
def add_to_cart():
    data = request.get_json()
    
    # Check if product exists
    product = Product.query.get(data['product_id'])
    if not product:
        return jsonify({'message': 'Product not found'}), 404
    
    # الحصول على وقت التفاعل الحالي
    current_time = dt.now()

    # Get or create cart for user
    cart = Cart.query.filter_by(user_id=data['user_id']).first()
    if not cart:
        cart = Cart(user_id= data['user_id'], created_at=dt.now())
        db.session.add(cart)
        db.session.commit()
    
    # Check if product already in cart
    cart_item = CartItem.query.filter_by(cart_id=cart.cart_id, product_id=data['product_id']).first()
    
    if cart_item:
        cart_item.quantity += data.get('quantity', 1)
        interaction_type = 'update_cart'
    else:
        cart_item = CartItem(
            cart_id=cart.cart_id,
            product_id=data['product_id'],
            quantity=data.get('quantity', 1)
        )
        db.session.add(cart_item)
        interaction_type = 'add_to_cart'
    
    db.session.commit()

    # ========== إضافة البيانات إلى هياكل البيانات ==========
    global interactions
    global context
    # 1. إضافة إلى interactions DataFrame
    new_interaction = {
        'user_id': data['user_id'],
        'product_id': data['product_id'],
        'interaction_type': interaction_type,
        'timestamp': current_time.strftime('%m/%d/%Y %I:%M:%S %p')
    }
    
    interactions = pd.concat([interactions, pd.DataFrame([new_interaction])], ignore_index=True)
    
    # 2. إضافة إلى context DataFrame
    time_of_day = get_time_of_day(current_time)
    device = data.get('device', 'desktop')  # افتراضي desktop إذا لم يحدد
    location = data.get('location', 'unknown')  # افتراضي unknown إذا لم يحدد
    
    new_context = {
        'interaction_id': len(context) + 1,  # أو استخدام ID فريد آخر
        'time_of_day': time_of_day,
        'device': device,
        'location': location
    }
    
    context = pd.concat([context, pd.DataFrame([new_context])], ignore_index=True)
    
    # حفظ التغييرات في ملفات CSV 
    interactions.to_csv(os.path.join(DATA_DIR, 'interactions.csv'), index=False)
    context.to_csv(os.path.join(DATA_DIR, 'context.csv'), index=False)
    
    return jsonify({'message': 'Product added to cart successfully', 'cart': cart_schema.dump(cart)}), 201

def get_time_of_day(dt):
    hour = dt.hour
    if 5 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 17:
        return 'afternoon'
    elif 17 <= hour < 21:
        return 'evening'
    else:
        return 'night'

@app.route('/api/cart/item/<int:item_id>', methods=['DELETE'])
def remove_from_cart(item_id):
    cart_item = CartItem.query.get_or_404(item_id)
    cart = Cart.query.get_or_404(cart_item.cart_id)
    
  
    db.session.delete(cart_item)
    db.session.commit()
     
    return jsonify({'message': cart_item.cart_id})
 
@app.route('/api/cart/<int:user_id>', methods=['GET'])

def get_cart(user_id):
    
    cart = Cart.query.filter_by(user_id=user_id).first()
    if not cart:
        return jsonify({'message': 'Cart is empty'}), 404
    cart_data = {
        "cart": {
            "items": [{
                "cart_item_id": item.cart_item_id,
                "quantity": item.quantity,
                "product": {
                    "product_id": item.product.product_id,
                    "product_name": item.product.product_name,
                    "price": item.product.price,
                    "product_image": item.product.product_image,
                }
            } for item in cart.items]
        }
    }
    
    return jsonify({'cart': cart_schema.dump(cart)})

@app.route('/api/cart', methods=['DELETE'])
def clear_cart():
    data = request.get_json()
    cart = Cart.query.filter_by(user_id=data['user_id']).first()
    
    if cart:
            # حذف جميع عناصر السلة المرتبطة بهذه السلة
            CartItem.query.filter_by(cart_id=cart.cart_id).delete()
            
            # بدلاً من تعيين cart_id إلى null، نمسح العناصر مباشرة
            db.session.commit()
            return jsonify({'message': 'Cart cleared successfully'})
    else:
            return jsonify({'message': 'Cart is already empty'}), 404
   
    
    

@app.route('/api/cart/purchase', methods=['POST'])
def purchase_cart():
    data = request.get_json()
    
    # التحقق من وجود المستخدم والسلة
    cart = Cart.query.filter_by(user_id=data['user_id']).first()
    if not cart:
        return jsonify({'message': 'Cart not found'}), 404
    
    # الحصول على وقت الشراء الحالي
    current_time = dt.now()
    
    # ========== إضافة بيانات الشراء إلى هياكل البيانات ==========
    global interactions
    global context
    # 1. إضافة إلى interactions DataFrame لكل منتج في السلة
    cart_items = CartItem.query.filter_by(cart_id=cart.cart_id).all()
    
    for item in cart_items:
        # تسجيل تفاعل purchase لكل منتج
        new_interaction = {
            'user_id': data['user_id'],
            'product_id': item.product_id,
            'interaction_type': 'purchase',
            'timestamp': current_time.strftime('%m/%d/%Y %I:%M:%S %p')
        }
        interactions = pd.concat([interactions, pd.DataFrame([new_interaction])], ignore_index=True)
        
        # تسجيل سياق الشراء
        time_of_day = get_time_of_day(current_time)
        device = data.get('device', 'desktop')  # افتراضي desktop إذا لم يحدد
        location = data.get('location', 'unknown')  # افتراضي unknown إذا لم يحدد
        
        new_context = {
            'interaction_id': len(context) + 1,  # أو استخدام ID فريد آخر
            'time_of_day': time_of_day,
            'device': device,
            'location': location
        }
        context = pd.concat([context, pd.DataFrame([new_context])], ignore_index=True)
    
    # حفظ التغييرات في ملفات CSV 
    interactions.to_csv(os.path.join(DATA_DIR, 'interactions.csv'), index=False)
    context.to_csv(os.path.join(DATA_DIR, 'context.csv'), index=False)
    
    return jsonify({
        'message': 'The purchase process has been completed',
        'purchased_items': len(cart_items),
        'purchase_time': current_time.strftime('%m/%d/%Y %I:%M:%S %p')
    })
    

@app.route('/api/products/view', methods=['POST'])
def track_product_view():
    data = request.get_json()
    
    # التحقق من وجود المنتج
    product = Product.query.get(data['product_id'])
    if not product:
        return jsonify({'message': 'Product not found'}), 404
    
    # الحصول على وقت التفاعل الحالي
    current_time = dt.now()
    
    # ========== إضافة البيانات إلى هياكل البيانات ==========
    
    # 1. إضافة إلى interactions DataFrame
    new_interaction = {
        'user_id': data.get('user_id', 0),  # 0 إذا لم يكن مستخدم مسجل
        'product_id': data['product_id'],
        'interaction_type': 'view',
        'timestamp': current_time.strftime('%m/%d/%Y %I:%M:%S %p')
    }
    
    global interactions
    interactions = pd.concat([interactions, pd.DataFrame([new_interaction])], ignore_index=True)
    
    # 2. إضافة إلى context DataFrame
    time_of_day = get_time_of_day(current_time)
    device = data.get('device', request.user_agent.string.split('/')[0] if request.user_agent else 'desktop')
    location = data.get('location', 'unknown')

    global context
    new_context = {
        'interaction_id': len(context) + 1,
        'time_of_day': time_of_day,
        'device': device,
        'location': location
    }
    
    
    context = pd.concat([context, pd.DataFrame([new_context])], ignore_index=True)
    
    # حفظ التغييرات في ملفات CSV (اختياري)
    interactions.to_csv(os.path.join(DATA_DIR, 'interactions.csv'), index=False)
    context.to_csv(os.path.join(DATA_DIR, 'context.csv'), index=False)
    
    return jsonify({
        'message': 'Product view tracked successfully',
        'product_id': data['product_id'],
        'view_time': current_time.strftime('%Y-%m-%d %H:%M:%S')
    })



# Admin Routes
@app.route('/api/users', methods=['GET'])
@cache.cached(timeout=400, query_string=True)
def get_all_users():
   
    users = User.query.all()
    return jsonify({'users': users_schema.dump(users)})


# Admin Routes
@app.route('/api/number-of=interactions', methods=['GET'])
@cache.cached(timeout=400, query_string=True)
def get_number_uinteractions():
    number = len(interactions)
    return jsonify({'number': number})



### RECOMENDATION ENDPOINTS


@app.route("/api/image/recommend", methods=["POST"])
def recommend_by_image():
    """Get recommendations based on product image similarity"""
   
    try:
        data = request.get_json()
        recommendations = []
        
        image_recommendations = service.get_recommendations(data['image_name'], data.get('top_n', 3))
        for pro in image_recommendations:
            product = Product.query.get_or_404(pro['product_id'])
            recommendations.append(
                product_schema.dump(product)
            )
        return jsonify({"products" : recommendations})
    except Exception as e:
        raise HTTPException(description=str(e))


@app.route("/api/content-based/simple", methods=["POST"])
def content_based_recommendation():
    """Get content-based recommendations based on column and value"""
    
    try:
        data = request.get_json()
        column_name = data['column_name']
        recommendation = []
        recommendation_based = content_based_simple(
            data['column_name'],
            data['value'],
            products,
            data.get('features', ['product_id','product_name','category', 'brand', 'country_made'])
        )
        recommendation_titles = recommendation_based.tolist()
        for recom in recommendation_titles:
            product = Product.query.filter(getattr(Product, column_name) == recom).first()
            recommendation.append(
               product_schema.dump(product)
               
            )
        return jsonify({"products" : recommendation})
    except Exception as e:
        raise HTTPException(description=str(e))
    

@app.route("/api/most-popular/<int:top_n>", methods=["GET"])
@cache.cached(timeout=400, query_string=True)
def most_popular(top_n):
    """Get most popular products based on user ratings"""
    
    try:
        most = []
        popular_products = mostPopular(reviews, top_n)
        product_ids_list = popular_products['product_id'].tolist()
        for pro_id in product_ids_list:
            product = Product.query.get_or_404(pro_id)
            most.append(
                product_schema.dump(product)
            )
        return jsonify({"products" : most})
    except Exception as e:
        raise HTTPException(description=str(e))
    
@app.route("/api/feature-based/knn", methods=["POST"])
def feature_based_knn_recommendation():
    """Get recommendations based on features using KNN"""
   
    try:
        data = request.get_json()
        if data.get('entity_type', 'product') == "product":
            df = metadata['products']
            pro_num = pd.DataFrame(metadata['features']['product_numerical'])
            pro_cat = pd.DataFrame(metadata['features']['product_categorical'])
        else:  # user
            df = metadata['users']
            pro_num = pd.DataFrame(metadata['features']['user_numerical'])
            pro_cat = pd.DataFrame(metadata['features']['user_categorical'])
        recommendation = []
        features = pd.concat([pro_num, pro_cat], axis=1)
        recommendations_knn = get_recomend_with_features_knn(df, features, data['column_name'], data['value'])
        if data.get('entity_type', 'product') == "product":
            for p in recommendations_knn:
                product = Product.query.get_or_404(int(p.iat[0]))
                recommendation.append(
                    product_schema.dump(product)
                )
            return jsonify({"products" : recommendation})
        else:
            for p in recommendations_knn:
                user = User.query.get_or_404(int(p.iat[0]))
                recommendation.append(
                    user_schema.dump(user)
                )
            return jsonify({"users" : recommendation})
    except Exception as e:
        raise HTTPException(description=str(e))


@app.route("/api/content-based/cosine", methods=["POST"])
def content_based_cosine_recommendation():
    """Get content-based recommendations using cosine similarity"""
   
    try:
        data = request.get_json()
        df = metadata['products']
        recommendation =[]
        pro_num = pd.DataFrame(metadata['features']['product_numerical'])
        pro_cat = pd.DataFrame(metadata['features']['product_categorical'])
        features = pd.concat([pro_num, pro_cat], axis=1)
        recommendations_cosine = get_recomend_with_cosine(df, features, data['column_name'], data['value'])
        for p in recommendations_cosine['product_id']:
                product = Product.query.get_or_404(p)
                recommendation.append(
                    product_schema.dump(product)
                )
        return jsonify({"products" : recommendation})
        
    except Exception as e:
        raise HTTPException(description=str(e))
    
@app.route("/api/text-based/recommend", methods=["POST"])
def text_based_recommendation():
    """Get recommendations based on text description or review"""
   
    try:
        data = request.get_json()
        recommendations = []
        
        if data.get('df', 'product_df') == "product_df":
            recommendations_desc = get_recomend_with_describition(
                products, data['product_id'], data['id'], data['description_column']
            )
            
            for pid in recommendations_desc.tolist():
                # تحويل int64 إلى int عادي
                product_id = int(pid) if hasattr(pid, 'item') else pid
                product = Product.query.get_or_404(product_id)
                recommendations.append(product_schema.dump(product))
                
        return jsonify({"products": recommendations})
    except Exception as e:
        raise HTTPException(description=str(e))

@app.route("/api/recommend-describtion-user", methods = ["POST"])
def products_by_description_from_user():
    try:
        data = request.get_json()
        recommendations = []
        product_ids = suggest_products_by_description_from_user(
                            products_df= products,  
                            user_description= data['description'],
                            top_n=5 
                        )
        for pid in product_ids:
                # تحويل int64 إلى int عادي
                product_id = int(pid) if hasattr(pid, 'item') else pid
                product = Product.query.get_or_404(product_id)
                recommendations.append(product_schema.dump(product))
        return jsonify({"products": recommendations})
    except Exception as e:
        raise HTTPException(description=str(e))

@app.route("/api/cluster-data", methods=["POST"])
def cluster_data_endpoint():
    """Cluster data based on numerical features"""
   
    try:
        data = request.get_json()
        
        if data['df'] == 'users':
            clustered_data = cluster_data(
                users,
                features=data['features'],
                n_clusters=data.get('n_clusters', 10),
                auto_optimize_k=data.get('auto_optimize_k', False),
                max_k=data.get('max_k', 15)
            )
            result = clustered_data[['user_id'] + data['features'] + ['cluster']].to_dict(orient='records')
        elif data['df'] == 'products':
            clustered_data = cluster_data(
                products,
                features=data['features'],
                n_clusters=data.get('n_clusters', 10),
                auto_optimize_k=data.get('auto_optimize_k', False),
                max_k=data.get('max_k', 15)
            )
            # إرجاع بيانات المنتج الكاملة بدلاً من المعرفات فقط
            clustered_products = []
            for _, row in clustered_data.iterrows():
                # تحويل int64 إلى int عادي
                product_id = int(row['product_id']) if hasattr(row['product_id'], 'item') else row['product_id']
                product = Product.query.get_or_404(product_id)
                product_data = product_schema.dump(product)
                product_data['cluster'] = int(row['cluster']) if hasattr(row['cluster'], 'item') else row['cluster']
                clustered_products.append(product_data)
            result = clustered_products
        else:
            raise NotFound("Only users and products can be clustered")
            
        return jsonify(result)
    except Exception as e:
        raise HTTPException(description=str(e))

@app.route("/api/description-based/recommend", methods=["POST"])
def description_based_recommendation():
    """Get recommendations based on user description"""
   
    try:
        data = request.get_json()
        recommendations = []
        
        recommendations_desc = recommend_products_by_description(
            products, 
            user_description=data['text'],
            description_column=data.get('column_name', 'description'), 
            top_n=data.get('top_n', 5)
        )
        
        # التحقق مما إذا كانت recommendations_desc DataFrame فارغاً
        if isinstance(recommendations_desc, pd.DataFrame) and not recommendations_desc.empty:
            for pid in recommendations_desc['product_id'].tolist():
                # تحويل int64 إلى int عادي
                product_id = int(pid) if hasattr(pid, 'item') else pid
                product = Product.query.get_or_404(product_id)
                recommendations.append(product_schema.dump(product))
        elif isinstance(recommendations_desc, list):
            for pid in recommendations_desc:
                # تحويل int64 إلى int عادي
                product_id = int(pid) if hasattr(pid, 'item') else pid
                product = Product.query.get_or_404(product_id)
                recommendations.append(product_schema.dump(product))
            
        return jsonify({"products": recommendations})
    except Exception as e:
        raise HTTPException(description=str(e))

@app.route("/api/similar-users_features/<int:user_id>", methods=["GET"])
def get_similar_users_features(user_id):
    """Find users with similar interests to the target user"""
    try:
        similar_users = []
        similar_users_pre = get_similar_user_from_userfeatures(user_id=user_id)
        
        for user_id in similar_users_pre:
            user_id_int = int(user_id) if hasattr(user_id, 'item') else user_id
            user = User.query.get_or_404(user_id_int)
            similar_users.append(user_schema.dump(user))
            
        return jsonify({"users": similar_users})
    except Exception as e:
        raise HTTPException(description=str(e))

@app.route("/api/similar-users/<int:user_id>", methods=["GET"])
def get_similar_users(user_id):
    """Find users with similar interests to the target user"""
    try:
        n_recommendations = request.args.get('n_recommendations', default=5, type=int)
        
        users_with_interests = extract_user_interests(users, reviews, interactions, products)
        similar_users = find_similar_users(
            target_user_id=user_id,
            users_with_interests_df=users_with_interests,
            reviews_df=reviews,
            interactions_df=interactions,
            n_recommendations=n_recommendations
        )
        similar_users_data = []
        for _, row in similar_users.iterrows():
            user_id_val = int(row['user_id']) if hasattr(row['user_id'], 'item') else row['user_id']
            user = User.query.get_or_404(user_id_val)
            user_data = user_schema.dump(user)
            
            similarity_score = row['similarity_score']
            if hasattr(similarity_score, 'item'):
                similarity_score = similarity_score.item()
            elif isinstance(similarity_score, (np.float64, np.float32)):
                similarity_score = float(similarity_score)
                
            user_data['similarity_score'] = similarity_score
            similar_users_data.append(user_data)
         
        return jsonify({"users": similar_users_data})
    except Exception as e:
        raise HTTPException(description=str(e))

@app.route("/api/similarity-between-users/<int:user_id1>/<int:user_id2>", methods=["GET"])
def get_similarity_between_users(user_id1, user_id2):
    """Calculate similarity score between two specific users"""
    try:
        # استخراج بيانات اهتمامات المستخدمين
        users_with_interests = extract_user_interests(users, reviews, interactions, products)
        
        # البحث عن بيانات المستخدمين المحددين
        user1_data = users_with_interests[users_with_interests['user_id'] == user_id1].iloc[0]
        user2_data = users_with_interests[users_with_interests['user_id'] == user_id2].iloc[0]
        
        # تحويل البيانات إلى قاموس
        user1_dict = {
            'interests_categories': user1_data['interests_categories'],
            'preferred_brands': user1_data['preferred_brands'],
            'avg_preferred_price': user1_data['avg_preferred_price']
        }
        
        user2_dict = {
            'interests_categories': user2_data['interests_categories'],
            'preferred_brands': user2_data['preferred_brands'],
            'avg_preferred_price': user2_data['avg_preferred_price']
        }
        
        # حساب درجة التشابه
        similarity_score = calculate_similarity_score(user1_dict, user2_dict)
        
        return jsonify({
            "user_id1": user_id1,
            "user_id2": user_id2,
            "similarity_score": similarity_score
        })
        
    except Exception as e:
        raise HTTPException(description=str(e))
    
def calculate_similarity_score(user1_data, user2_data):
    """حساب درجة التشابه بين مستخدمين بناءً على اهتماماتهم"""
    score = 0.0
    
    # التشابه في الفئات
    if 'interests_categories' in user1_data and 'interests_categories' in user2_data:
        common_categories = set(user1_data['interests_categories']) & set(user2_data['interests_categories'])
        score += len(common_categories) * 0.3
    
    # التشابه في الماركات
    if 'preferred_brands' in user1_data and 'preferred_brands' in user2_data:
        common_brands = set(user1_data['preferred_brands']) & set(user2_data['preferred_brands'])
        score += len(common_brands) * 0.2
    
    # التشابه في نطاق السعر
    if 'avg_preferred_price' in user1_data and 'avg_preferred_price' in user2_data:
        price_diff = abs(user1_data['avg_preferred_price'] - user2_data['avg_preferred_price'])
        if price_diff < 50:  # إذا كان الفرق أقل من 50
            score += 0.2
    
    return min(score, 1.0)  # التأكد من أن النتيجة لا تتجاوز 1.0
@app.route('/api/user-high-rated/<int:user_id>', methods=['GET'])
def get_user_high_rated_products(user_id):
    """Get products rated highly by a user"""
   
    try:
        min_rating = request.args.get('min_rating', default=3, type=int)
        high_rated_products = []
        
        high_rated = get_high_rated_products(reviews, user_id, min_rating)
        
        for product_id in high_rated:
            # تحويل int64 إلى int عادي
            product_id_int = int(product_id) if hasattr(product_id, 'item') else product_id
            product = Product.query.get_or_404(product_id_int)
            high_rated_products.append(product_schema.dump(product))
            
        return jsonify({
            "user_id": user_id,
            "min_rating": min_rating,
            "high_rated_products": high_rated_products
        })
    except Exception as e:
        raise HTTPException(description=str(e))

@app.route('/api/user-interacted-products/<int:user_id>', methods=['GET'])
def get_user_interacted_products_endpoint(user_id):
    """Get products a user has interacted with"""
    
    try:
        interacted_products_list = []
        
        interacted_products = get_user_interacted_products(interactions, user_id)
        
        for product_id in interacted_products:
            # تحويل int64 إلى int عادي
            product_id_int = int(product_id) if hasattr(product_id, 'item') else product_id
            product = Product.query.get_or_404(product_id_int)
            interacted_products_list.append(product_schema.dump(product))
            
        return jsonify({
            "user_id": user_id,
            "interacted_products": interacted_products_list
        })
    except Exception as e:
        raise HTTPException(description=str(e))

@app.route('/api/user-preferred-products/<int:user_id>', methods=['GET'])
def get_user_preferred_products(user_id):
    """Get products a user has interacted with or rated highly"""
   
    try:
        preferred_products_list = []
        
        preferred_products = get_user_prefered_product(reviews, interactions, user_id)
        
        for product_id in preferred_products:
            # تحويل int64 إلى int عادي
            product_id_int = int(product_id) if hasattr(product_id, 'item') else product_id
            product = Product.query.get_or_404(product_id_int)
            preferred_products_list.append(product_schema.dump(product))
            
        return jsonify({
            "user_id": user_id,
            "preferred_products": preferred_products_list
        })
    except Exception as e:
        raise HTTPException(description=str(e))

@app.route('/api/user-interactions-details/<int:user_id>', methods=['GET'])
def get_user_interactions_details_endpoint(user_id):
    """Get all interaction details for a user"""
    
    try:
        interactions_details = []
        
        interaction = get_user_interactions_details(interactions, user_id)
        
        for _, interaction in interaction.iterrows():
            # تحويل int64 إلى int عادي
            product_id = int(interaction['product_id']) if hasattr(interaction['product_id'], 'item') else interaction['product_id']
            product = Product.query.get_or_404(product_id)
            
            interaction_data = {}
            for key, value in interaction.to_dict().items():
                if hasattr(value, 'item'):
                    interaction_data[key] = value.item()
                elif isinstance(value, (np.int64, np.int32, np.float64, np.float32)):
                    interaction_data[key] = float(value) if isinstance(value, (np.float64, np.float32)) else int(value)
                else:
                    interaction_data[key] = value
                    
            interaction_data['product'] = product_schema.dump(product)
            interactions_details.append(interaction_data)
            
        return jsonify({
            "user_id": user_id,
            "interactions": interactions_details
        })
    except Exception as e:
        raise HTTPException(description=str(e))

@app.route('/api/user-interactions-by-type/<int:user_id>', methods=['GET'])
def get_user_interactions_by_type_endpoint(user_id):
    """Get user interactions filtered by type"""
    
    try:
        interaction_type = request.args.get('interaction_type', type=str)
        interactions_list = []
        
        interaction = get_user_interactions_by_type(interactions, user_id, interaction_type)
        
        for product_id in interaction:
            # تحويل int64 إلى int عادي
            product_id_int = int(product_id) if hasattr(product_id, 'item') else product_id
            product = Product.query.get_or_404(product_id_int)
            interactions_list.append(product_schema.dump(product))
            
        return jsonify({
            "user_id": user_id,
            "interaction_type": interaction_type,
            "interactions": interactions_list
        })
    except Exception as e:
        raise HTTPException(description=str(e))

@app.route('/api/similar-users-products/<int:user_id>', methods=['GET'])
def get_similar_users_products(user_id):
    """Get products preferred by users with similar interests"""
    
    try:
        products_list = []
        
        prods = get_user_similar_prefered_products(
            user_id, users, products, interactions, reviews
        )
        
        for product_id in prods:
            # تحويل int64 إلى int عادي
            product_id_int = int(product_id) if hasattr(product_id, 'item') else product_id
            product = Product.query.get_or_404(product_id_int)
            products_list.append(product_schema.dump(product))
            
        return jsonify({
            "user_id": user_id,
            "products_from_similar_users": products_list
        })
    except Exception as e:
        raise HTTPException(description=str(e))

@app.route('/api/product-based-recommendations', methods=['POST'])
def get_product_based_recommendations():
    """Get recommendations based on specific products"""
    try:
        data = request.get_json()
        product_ids = data.get('product_ids', [])
        
        # استخدام الكاش إذا كان متاحاً
        cached_recommendations = get_cached_products_recommendations(product_ids)
        if cached_recommendations:
            recommendations_list = []
            all_recommended_products = set()
            
            for key, product_list in cached_recommendations.items():
                if isinstance(product_list, list):
                    all_recommended_products.update(product_list)
            
            for product_id in all_recommended_products:
                product_id_int = int(product_id) if hasattr(product_id, 'item') else product_id
                product = Product.query.get(product_id_int)
                if product:
                    recommendations_list.append(product_schema.dump(product))
            
            return jsonify({
                "product_ids": product_ids,
                "source": "cache",
                "recommendations": recommendations_list
            })
        
        # الطريقة الأصلية
        recommendations_list = []
        recommendations = get_products_from_product(product_ids)
        
        all_recommended_products = set()
        for key, product_list in recommendations.items():
            if isinstance(product_list, list):
                all_recommended_products.update(product_list)
        
        for product_id in all_recommended_products:
            product_id_int = int(product_id) if hasattr(product_id, 'item') else product_id
            product = Product.query.get_or_404(product_id_int)
            recommendations_list.append(product_schema.dump(product))
            
        return jsonify({
            "product_ids": product_ids,
            "source": "live",
            "recommendations": recommendations_list
        })
    except Exception as e:
        raise HTTPException(description=str(e))

@app.route('/api/user-suggestions/<int:user_id>', methods=['GET'])
def get_user_suggestions(user_id):
    """Get comprehensive suggestions for a user"""
    try:
        suggestions_list = []
        suggestions = suggest_for_user(user_id)
        
        all_suggested_products = set()
        
        if 'user_products_based' in suggestions:
            for key, product_list in suggestions['user_products_based'].items():
                if isinstance(product_list, list):
                    all_suggested_products.update(product_list)
        
        if 'similar_users_based' in suggestions and isinstance(suggestions['similar_users_based'], list):
            all_suggested_products.update(suggestions['similar_users_based'])
        
        for product_id in all_suggested_products:
            product_id_int = int(product_id) if hasattr(product_id, 'item') else product_id
            product = Product.query.get_or_404(product_id_int)
            suggestions_list.append(product_schema.dump(product))
            
        return jsonify({
            "user_id": user_id,
            "source": "live",
            "suggestions": suggestions_list
        })
    except Exception as e:
        raise HTTPException(description=str(e))


@app.route('/api/deep-user-suggestions/<int:user_id>', methods=['GET'])
def get_deep_user_suggestions(user_id):
    """Get deep analysis suggestions for a user with predicted ratings"""
    
    try:
        suggestions_list = []
        
        suggestions = deep_suggest_for_user(user_id)
        
        for product_id, predicted_rating in suggestions.items():
            # تحويل int64 إلى int عادي
            product_id_int = int(product_id) if hasattr(product_id, 'item') else product_id
            
            product = Product.query.get_or_404(product_id_int)
            product_data = product_schema.dump(product)
            
            # تحويل numpy types إلى Python types
            if hasattr(predicted_rating, 'item'):
                predicted_rating = predicted_rating.item()
            elif isinstance(predicted_rating, (np.float64, np.float32)):
                predicted_rating = float(predicted_rating)
                
            product_data['predicted_rating'] = predicted_rating
            suggestions_list.append(product_data)
            
        return jsonify({
            "user_id": user_id,
            "suggestions": suggestions_list
        })
    except Exception as e:
        raise HTTPException(description=str(e))


@app.route("/api/user-interests/<int:user_id>", methods=["GET"])
def get_user_interests(user_id):
    """Extract user interests from their data"""
    try:
        users_with_interests = extract_user_interests(users, reviews, interactions, products)
        user_data = users_with_interests[users_with_interests['user_id'] == user_id].to_dict(orient='records')
        if not user_data:
            raise NotFound("User not found")
        return jsonify(user_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# معالجة الأخطاء
@app.errorhandler(404)
def not_found(e):
    return jsonify(error=str(e)), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify(error=str(e)), 500


@app.errorhandler(422)
def handle_unprocessable_entity(err):
    headers = err.data.get("headers", None)
    messages = err.data.get("messages", ["Invalid request"])
    app.logger.error(f'Unprocessable Entity: {messages}')
    return jsonify({
        "success": False,
        "error": 422,
        "messages": messages
    }), 422

@app.teardown_appcontext
def shutdown_session(exception=None):
    db_session.remove()

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

