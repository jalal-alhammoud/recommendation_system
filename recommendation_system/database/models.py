
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash
from flask_bcrypt import Bcrypt

bcrypt = Bcrypt()
db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    
    user_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_name = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    age = db.Column(db.Integer)
    gender = db.Column(db.String(20))
    location = db.Column(db.String(100))
    user_image = db.Column(db.String(200))
    admin = db.Column(db.Boolean, default=False)
    
    reviews = db.relationship('Review', backref='user', lazy=True)
    carts = db.relationship('Cart', backref='user', lazy=True)

    def __init__(self, user_name, password, email, age=None, gender=None, location=None, user_image=None, admin=False):
        self.user_name = user_name
        self.set_password(password)
        self.email = email
        self.age = age
        self.gender = gender
        self.location = location
        self.user_image = user_image
        self.admin = admin
    
    def set_password(self, password):
        self.password = bcrypt.generate_password_hash(password).decode('utf-8')
    
    def check_password(self, password):
        return bcrypt.check_password_hash(self.password, password)
    
class Category(db.Model):
    __tablename__ = 'categories'
    
    cat_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    cat_name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    cat_image = db.Column(db.String(200))

    products = db.relationship('Product', backref='category', lazy=True)

class Product(db.Model):
    __tablename__ = 'products'
    
    product_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    product_name = db.Column(db.String(500), nullable=False)
    price = db.Column(db.Float, nullable=False)
    cat_id = db.Column(db.Integer, db.ForeignKey('categories.cat_id'), nullable=False)
    product_image = db.Column(db.String(500))
    description = db.Column(db.Text)
    country_made = db.Column(db.String(50))
    brand = db.Column(db.String(50))
    
    reviews = db.relationship('Review', backref='product', lazy=True)
    cart_items = db.relationship('CartItem', backref='product', lazy=True)

class Review(db.Model):
    __tablename__ = 'reviews'
    
    review_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey('products.product_id'), nullable=False)
    rating = db.Column(db.Integer, nullable=False)
    review_text = db.Column(db.Text)

class Cart(db.Model):
    __tablename__ = 'carts'
    
    cart_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False)
    
    items = db.relationship('CartItem', backref='cart', lazy=True)

class CartItem(db.Model):
    __tablename__ = 'cart_items'
    
    cart_item_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    cart_id = db.Column(db.Integer, db.ForeignKey('carts.cart_id'), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey('products.product_id'), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)

   