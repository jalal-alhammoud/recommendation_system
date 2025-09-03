/**
 * API Service للتواصل مع الواجهة الخلفية
 * يتضمن جميع الوظائف اللازمة للتعامل مع نقاط النهاية المختلفة
 */

class APIService {
    constructor() {
        this.baseURL = 'http://127.0.0.1:5000/api';
        this.authToken =  '';
    }

    /**
     * تحديث إعدادات API
     * @param {string} baseURL - رابط الواجهة الخلفية
     * @param {string} authToken - رمز المصادقة (اختياري)
     */
    updateConfig(baseURL, authToken = '') {
        this.baseURL = baseURL;
        this.authToken = authToken;
        
        // حفظ الإعدادات في localStorage
        localStorage.setItem('apiBaseURL', baseURL);
        if (authToken) {
            localStorage.setItem('apiAuthToken', authToken);
        }
    }

    /**
     * تنفيذ طلب API
     * @param {string} url - الرابط النهائي
     * @param {string} method - طريقة الطلب (GET, POST, etc.)
     * @param {Object} data - بيانات الطلب (اختياري)
     * @param {boolean} isFormData - هل البيانات من نوع FormData
     * @returns {Promise} - وعد بنتيجة الطلب
     */
    async _request(url, method = 'GET', data = null, isFormData = false) {
        const options = {
            method: method,
            headers: {}
        };

        // إضافة رمز المصادقة إذا كان متوفراً
        if (this.authToken) {
            options.headers['Authorization'] = `Bearer ${this.authToken}`;
        }

        // إضافة البيانات للطلب حسب نوعه
        if (data && method !== 'GET') {
            if (isFormData) {
                options.body = data;
                // لا نضيف Content-Type للـ FormData، المتصفح سيفعل ذلك تلقائياً
            } else {
                options.headers['Content-Type'] = 'application/json';
                options.body = JSON.stringify(data);
            }
        } else if (data && method === 'GET') {
            // لطلبات GET، نضيف المعلمات إلى URL كاستعلام
            const params = new URLSearchParams(data);
            url += (url.includes('?') ? '&' : '?') + params.toString();
        }

        try {
            const response = await fetch(`${this.baseURL}${url}`, options);
            const responseText = await response.text();
            
            // التحقق مما إذا كانت الاستجابة هي HTML (خطأ)
            if (responseText.trim().startsWith('<!')) {
                throw new Error(`الخادم أعاد HTML بدلاً من JSON. قد يكون هناك خطأ في الخادم أو في البيانات المرسلة`);
            }
            
            const result = JSON.parse(responseText);
            
            if (!response.ok) {
                throw new Error(`خطأ في الطلب: ${response.status} ${response.statusText} - ${JSON.stringify(result)}`);
            }
            
            return result;
        } catch (error) {
            console.error('Error in API request:', error);
            throw error;
        }
    }

  
    /**
     * تسجيل مستخدم جديد
     * @param {Object} userData - بيانات المستخدم
     * @param {string} userData.username - اسم المستخدم
     * @param {string} userData.email - البريد الإلكتروني
     * @param {string} userData.password - كلمة المرور
     * @param {number} userData.age - العمر (اختياري)
     * @param {string} userData.gender - الجنس (اختياري)
     * @param {string} userData.location - الموقع (اختياري)
     * @param {File} userData.user_image - صورة المستخدم (اختياري)
     * @returns {Promise} - وعد بنتيجة التسجيل
     */
    async signup(userData) {
        const formData = new FormData();
        
        // إضافة الحقول النصية
        Object.keys(userData).forEach(key => {
            if (key !== 'user_image' && userData[key] !== undefined) {
                formData.append(key, userData[key]);
            }
        });
        
        // إضافة ملف الصورة إذا وجد
        if (userData.user_image) {
            formData.append('user_image', userData.user_image);
        }
        
        return this._request('/signup', 'POST', formData, true);
    }

    /**
     * تسجيل الدخول
     * @param {string} identifier - اسم المستخدم أو البريد الإلكتروني
     * @param {string} password - كلمة المرور
     * @returns {Promise} - وعد بنتيجة تسجيل الدخول
     */
    async login(identifier, password) {
        return this._request('/login', 'POST', { identifier, password });
    }

   

    /**
     * الحصول على بيانات المستخدم
     * @param {number} id - معرف المستخدم
     * @returns {Promise} - وعد ببيانات المستخدم
     */
    async getUser(id) {
        return this._request(`/user/${id}`, 'GET');
    }

    /**
     * الحصول على بيانات كل المستخدمين
   
     * @returns {Promise} - وعد ببيانات المستخدم
     */
    async getAllUser() {
        return this._request(`/users`, 'GET');
    }

     /**
     * الحصول على عدد التفاعلات
   
     * @returns {Promise} - وعد ببيانات المستخدم
     */
    async getInterNumber() {
        return this._request(`/number-of=interactions`, 'GET');
    }

     /**
     * الحصول على كل ال brands
   
     * @returns {Promise} - وعد ببيانات المستخدم
     */
    async geAllBrands() {
        return this._request(`/brands`, 'GET');
    }
    
    /**
     * تحديث بيانات المستخدم
     * @param {Object} userData - بيانات المستخدم للتحديث
     * @param {number} userData.user_id - معرف المستخدم
     * @param {string} userData.username - اسم المستخدم (اختياري)
     * @param {string} userData.email - البريد الإلكتروني (اختياري)
     * @param {number} userData.age - العمر (اختياري)
     * @param {string} userData.gender - الجنس (اختياري)
     * @param {string} userData.location - الموقع (اختياري)
     * @param {File} userData.user_image - صورة المستخدم (اختياري)
     * @returns {Promise} - وعد بنتيجة التحديث
     */
    async updateUser(userData) {
        const formData = new FormData();
        
        // إضافة الحقول النصية
        Object.keys(userData).forEach(key => {
            if (key !== 'user_image' && userData[key] !== undefined) {
                formData.append(key, userData[key]);
            }
        });
        
        // إضافة ملف الصورة إذا وجد
        if (userData.user_image) {
            formData.append('user_image', userData.user_image);
        }
        
        return this._request('/user/updateprofile', 'POST', formData, true);
    }


    /**
     * الحصول على جميع المنتجات
     * @param {number} limit - الحد الأقصى لعدد المنتجات (اختياري)
     * @returns {Promise} - وعد بقائمة المنتجات
     */
    async getProducts(limit = null) {
        const params = {};
        if (limit) params.limit = limit;
        
        return this._request('/products', 'GET', params);
    }

    /**
     * الحصول على منتج محدد
     * @param {number} id - معرف المنتج
     * @returns {Promise} - وعد ببيانات المنتج
     */
    async getProduct(id) {
        return this._request(`/products/${id}`, 'GET');
    }

    /**
     * بحث المنتجات
     * @param {string} query - كلمة البحث
     * @returns {Promise} - وعد بنتائج البحث
     */
    async searchProducts(query) {
        return this._request('/products/search', 'GET', { q: query });
    }

    /**
     * المنتجات حسب الفئة
     * @param {number} categoryId - معرف الفئة
     * @returns {Promise} - وعد بالمنتجات في الفئة المحددة
     */
    async getProductsByCategory(categoryId) {
        return this._request(`/products/category/${categoryId}`, 'GET');
    }

    /**
     * المنتجات حسب البلد
     * @param {string} country - اسم البلد
     * @returns {Promise} - وعد بالمنتجات من البلد المحدد
     */
    async getProductsByCountry(country) {
        return this._request(`/products/country/${country}`, 'GET');
    }

    /**
     * المنتجات حسب الماركة
     * @param {string} brand - اسم الماركة
     * @returns {Promise} - وعد بالمنتجات من الماركة المحددة
     */
    async getProductsByBrand(brand) {
        return this._request(`/products/brand/${brand}`, 'GET');
    }

    /**
     * إضافة منتج جديد
     * @param {Object} productData - بيانات المنتج
     * @param {string} productData.product_name - اسم المنتج
     * @param {string} productData.description - وصف المنتج
     * @param {number} productData.price - سعر المنتج
     * @param {string} productData.country_made - بلد المنشأ
     * @param {string} productData.brand - الماركة
     * @param {number} productData.cat_id - معرف الفئة
     * @param {File} productData.product_image - صورة المنتج (اختياري)
     * @returns {Promise} - وعد بنتيجة الإضافة
     */
    async addProduct(productData) {
        const formData = new FormData();
        
        // إضافة الحقول النصية
        Object.keys(productData).forEach(key => {
            if (key !== 'product_image' && productData[key] !== undefined) {
                formData.append(key, productData[key]);
            }
        });
        
        // إضافة ملف الصورة إذا وجد
        if (productData.product_image) {
            formData.append('product_image', productData.product_image);
        }
        
        return this._request('/products', 'POST', formData, true);
    }

   
    /**
     * الحصول على جميع الفئات
     * @returns {Promise} - وعد بقائمة الفئات
     */
    async getCategories() {
        return this._request('/categories', 'GET');
    }

    /**
     * الحصول على فئة محددة
     * @param {number} id - معرف الفئة
     * @returns {Promise} - وعد ببيانات الفئة
     */
    async getCategory(id) {
        return this._request(`/categories/${id}`, 'GET');
    }

    /**
     * إضافة فئة جديدة
     * @param {Object} categoryData - بيانات الفئة
     * @param {string} categoryData.Cat_name - اسم الفئة
     * @param {string} categoryData.Description - وصف الفئة (اختياري)
     * @param {File} categoryData.Cat_image - صورة الفئة (اختياري)
     * @returns {Promise} - وعد بنتيجة الإضافة
     */
    async addCategory(categoryData) {
        const formData = new FormData();
        
        // إضافة الحقول النصية
        Object.keys(categoryData).forEach(key => {
            if (key !== 'Cat_image' && categoryData[key] !== undefined) {
                formData.append(key, categoryData[key]);
            }
        });
        
        // إضافة ملف الصورة إذا وجد
        if (categoryData.Cat_image) {
            formData.append('Cat_image', categoryData.Cat_image);
        }
        
        return this._request('/categories', 'POST', formData, true);
    }

    /**
     * تحديث فئة
     * @param {Object} categoryData - بيانات الفئة للتحديث
     * @param {number} categoryData.Cat_id - معرف الفئة
     * @param {string} categoryData.Cat_name - اسم الفئة (اختياري)
     * @param {string} categoryData.Description - وصف الفئة (اختياري)
     * @param {File} categoryData.Cat_image - صورة الفئة (اختياري)
     * @returns {Promise} - وعد بنتيجة التحديث
     */
    async updateCategory(categoryData) {
        const formData = new FormData();
        
        // إضافة الحقول النصية
        Object.keys(categoryData).forEach(key => {
            if (key !== 'Cat_image' && categoryData[key] !== undefined) {
                formData.append(key, categoryData[key]);
            }
        });
        
        // إضافة ملف الصورة إذا وجد
        if (categoryData.Cat_image) {
            formData.append('Cat_image', categoryData.Cat_image);
        }
        
        return this._request('/categories/update_category', 'POST', formData, true);
    }

   
    /**
     * إضافة تقييم
     * @param {Object} reviewData - بيانات التقييم
     * @param {number} reviewData.user_id - معرف المستخدم
     * @param {number} reviewData.product_id - معرف المنتج
     * @param {number} reviewData.rating - التقييم (1-5)
     * @param {string} reviewData.review_text - نص التقييم (اختياري)
     * @returns {Promise} - وعد بنتيجة الإضافة
     */
    async addReview(reviewData) {
        return this._request('/reviews', 'POST', reviewData);
    }

    /**
     * تقييمات المستخدم
     * @param {number} userId - معرف المستخدم
     * @returns {Promise} - وعد بتقييمات المستخدم
     */
    async getUserReviews(userId) {
        return this._request(`/reviews/user/${userId}`, 'GET');
    }

    /**
     * تقييمات المنتج
     * @param {number} productId - معرف المنتج
     * @returns {Promise} - وعد بتقييمات المنتج
     */
    async getProductReviews(productId) {
        return this._request(`/reviews/product/${productId}`, 'GET');
    }

   
    /**
     * إضافة إلى عربة التسوق
     * @param {Object} cartData - بيانات عربة التسوق
     * @param {number} cartData.user_id - معرف المستخدم
     * @param {number} cartData.product_id - معرف المنتج
     * @param {number} cartData.quantity - الكمية
     * @returns {Promise} - وعد بنتيجة الإضافة
     */
    async addToCart(cartData) {
        return this._request('/cart', 'POST', cartData);
    }

    /**
     * الحصول على عربة التسوق
     * @param {number} userId - معرف المستخدم
     * @returns {Promise} - وعد بمحتويات عربة التسوق
     */
    async getCart(userId) {
        return this._request(`/cart/${userId}`, 'GET');
    }

    /**
     * شراء محتويات عربة التسوق
     * @param {number} userId - معرف المستخدم
     * @returns {Promise} - وعد بنتيجة الشراء
     */
    async purchaseCart(userId) {
        return this._request('/cart/purchase', 'POST', { user_id: userId });
    }

  
    /**
     * توصيات SVD
     * @param {number} userId - معرف المستخدم
     * @param {number} productId - معرف المنتج
     * @returns {Promise} - وعد بالتوصيات
     */
    async getSVDRecommendations(userId, productId) {
        return this._request('/svd/recommend', 'POST', { user_id: userId, product_id: productId });
    }

    /**
     * توصيات KNN
     * @param {number} userId - معرف المستخدم
     * @param {number} productId - معرف المنتج
     * @returns {Promise} - وعد بالتوصيات
     */
    async getKNNRecommendations(userId, productId) {
        return this._request('/knn/recommend', 'POST', { user_id: userId, product_id: productId });
    }

    /**
     * التوصيات بالتعلم العميق
     * @param {Object} data - بيانات التوصية
     * @param {number} data.age - العمر
     * @param {string} data.gender - الجنس
     * @param {number} data.price - السعر
     * @param {string} data.location_user - موقع المستخدم
     * @param {string} data.review_text - نص المراجعة (اختياري)
     * @param {string} data.category - الفئة
     * @param {string} data.brand - الماركة
     * @returns {Promise} - وعد بالتوصيات
     */
    async getDeepLearningRecommendations(data) {
         const enhancedData = {
        ...data,
        user_id: 101,       // إضافة user_id ثابت
        product_id: 531227  // إضافة product_id ثابت
    };
        return this._request('/deeplearning/recommend', 'POST', enhancedData);
    }

    /**
     * التوصيات المعتمدة على المستخدم
     * @param {number} userId - معرف المستخدم
     * @returns {Promise} - وعد بالتوصيات
     */
    async getSimilarUsersProducts(userId) {
        return this._request(`/similar-users-products/${userId}`, 'GET');
    }

    /**
     * التوصيات المعتمدة على المنتج
     * @param {number|Array} productIds - معرف المنتج أو مجموعة معرفات
     * @returns {Promise} - وعد بالتوصيات
     */
    async getProductBasedRecommendations(productIds) {
        // تحويل إلى مصفوفة إذا كان معرفًا واحدًا
        const ids = Array.isArray(productIds) ? productIds : [productIds];
        return this._request('/product-based-recommendations', 'POST', { product_ids: ids });
    }

    /**
     * التوصيات بالتصفية التعاونية
     * @param {number} userId - معرف المستخدم
     * @returns {Promise} - وعد بالتوصيات
     */
    async getUserSuggestions(userId) {
        return this._request(`/user-suggestions/${userId}`, 'GET');
    }

    /**
     * التوصيات بناءً على وصف المستخدم
     * @param {string} description - وصف المستخدم
     * @returns {Promise} - وعد بالتوصيات
     */
    async getDescriptionBasedRecommendations(description) {
        return this._request('/recommend-describtion-user', 'POST', { description });
    }

    /**
     * التوصيات باستخدام KNN بناءً على الميزات
     * @param {Object} data - بيانات التوصية
     * @param {string} data.entity_type - نوع الكيان (product أو user)
     * @param {string} data.column_name - اسم العمود
     * @param {string} data.value - القيمة
     * @param {Array} data.features - الميزات (صفات التشابه)
     * @returns {Promise} - وعد بالتوصيات
     */
    async getFeatureBasedKNNRecommendations(data) {
        return this._request('/feature-based/knn', 'POST', data);
    }

    /**
     * التوصيات باستخدام cosine similarity بناءً على المحتوى
     * @param {Object} data - بيانات التوصية
     * @param {string} data.column_name - اسم العمود
     * @param {string} data.value - القيمة
     * @param {Array} data.features - الميزات (صفات التشابه)
     * @returns {Promise} - وعد بالتوصيات
     */
    async getContentBasedCosineRecommendations(data) {
        return this._request('/content-based/cosine', 'POST', data);
    }

    /**
     * المنتجات الأكثر شيوعاً
     * @param {number} n - عدد المنتجات المطلوبة
     * @returns {Promise} - وعد بالمنتجات الأكثر شيوعاً
     */
    async getMostPopularProducts(n) {
        return this._request(`/most-popular/${n}`, 'GET');
    }

    /**
     * التوصيات بناءً على الصورة
     * @param {string} imageName - اسم الصورة
     * @param {number} topN - عدد التوصيات المطلوبة (اختياري، افتراضي 3)
     * @returns {Promise} - وعد بالتوصيات
     */
    async getImageBasedRecommendations(imageName, topN = 3) {
        return this._request('/image/recommend', 'POST', { image_name: imageName, top_n: topN });
    }

   
    /**
     * اهتمامات المستخدم
     * @param {number} userId - معرف المستخدم
     * @returns {Promise} - وعد باهتمامات المستخدم
     */
    async getUserInterests(userId) {
        return this._request(`/user-interests/${userId}`, 'GET');
    }

    /**
     * تجميع البيانات
     * @param {Object} data - بيانات التجميع
     * @param {string} data.df - نوع البيانات (users أو products)
     * @param {Array} data.features - الميزات
     * @param {number} data.n_clusters - عدد المجموعات (اختياري، افتراضي 5)
     * @param {boolean} data.auto_optimize_k - تحسين تلقائي (اختياري، افتراضي false)
     * @param {number} data.max_k - الحد الأقصى للمجموعات (اختياري، افتراضي 15)
     * @returns {Promise} - وعد بنتيجة التجميع
     */
    async clusterData(data) {
        return this._request('/cluster-data', 'POST', data);
    }

    /**
     * تفاعلات المستخدم حسب النوع
     * @param {number} userId - معرف المستخدم
     * @param {string} interactionType - نوع التفاعل (add_to_cart || view || purchase)
     * @returns {Promise} - وعد بتفاصيل التفاعلات
     */
    async getUserInteractionsByType(userId, interactionType) {
        return this._request(`/user-interactions-by-type/${userId}`, 'GET', { interaction_type: interactionType });
    }

    /**
     * تفاصيل تفاعلات المستخدم
     * @param {number} userId - معرف المستخدم
     * @returns {Promise} - وعد بتفاصيل التفاعلات
     */
    async getUserInteractionsDetails(userId) {
        return this._request(`/user-interactions-details/${userId}`, 'GET');
    }

    /**
     * المنتجات المفضلة للمستخدم
     * @param {number} userId - معرف المستخدم
     * @returns {Promise} - وعد بالمنتجات المفضلة
     */
    async getUserPreferredProducts(userId) {
        return this._request(`/user-preferred-products/${userId}`, 'GET');
    }

    /**
     * المنتجات التي تفاعل معها المستخدم
     * @param {number} userId - معرف المستخدم
     * @returns {Promise} - وعد بالمنتجات التي تفاعل معها المستخدم
     */
    async getUserInteractedProducts(userId) {
        return this._request(`/user-interacted-products/${userId}`, 'GET');
    }

    /**
     * المنتجات الأعلى تقييماً من المستخدم
     * @param {number} userId - معرف المستخدم
     * @param {number} minRating - الحد الأدنى للتقييم (اختياري، افتراضي 3)
     * @returns {Promise} - وعد بالمنتجات الأعلى تقييماً
     */
    async getUserHighRatedProducts(userId, minRating = 3) {
        return this._request(`/user-high-rated/${userId}`, 'GET', { min_rating: minRating });
    }

    /**
     * المستخدمون المتشابهون
     * @param {number} userId - معرف المستخدم
     * @param {number} nRecommendations - عدد التوصيات (اختياري، افتراضي 5)
     * @returns {Promise} - وعد بالمستخدمين المتشابهين
     */
    async getSimilarUsers(userId, nRecommendations = 5) {
        return this._request(`/similar-users/${userId}`, 'GET', { n_recommendations: nRecommendations });
    }

    /**
     * المستخدمون المتشابهون (بناءً على الميزات)
     * @param {number} userId - معرف المستخدم
     * @returns {Promise} - وعد بالمستخدمين المتشابهين
     */
    async getSimilarUsersByFeatures(userId) {
        return this._request(`/similar-users_features/${userId}`, 'GET');
    }

    /**
     * حساب التشابه بين مستخدمين محددين
     * @param {number} userId1 - معرف المستخدم الأول
     * @param {number} userId2 - معرف المستخدم الثاني
     * @returns {Promise} - وعد بنتيجة التشابه
     */
    async getSimilarityBetweenUsers(userId1, userId2) {
        return this._request(`/similarity-between-users/${userId1}/${userId2}`, 'GET');
    }
}

window.APIService = new APIService();
