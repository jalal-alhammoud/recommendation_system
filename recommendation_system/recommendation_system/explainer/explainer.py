from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import logging
import torch
import re


class RecommendationExplainer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            USE_GPU= False
            # Device configuration with availability check
            device = 0 if USE_GPU and torch.cuda.is_available() else -1
            
            # Load model and tokenizer separately for better control
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            model = AutoModelForCausalLM.from_pretrained("gpt2")
            
            # Configure special tokens
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Create pipeline with optimized settings
            self.explainer = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=device,
                torch_dtype=torch.float16 if USE_GPU and torch.cuda.is_available() else torch.float32
            )
            
            # Enhanced list of unwanted phrases
            self.bad_words = [
                "email", "contact us", "follow us", "copyright", "©", 
                "twitter", "log in", "sales@", "website", "promotion",
                "discount", "click here", "subscribe", "call us"
            ]
            self.bad_words_ids = [tokenizer.encode(word, add_special_tokens=False) for word in self.bad_words]
            
            # Generation configuration
            self.generation_config = {
                'max_new_tokens': 120,  # Reduced length
                'temperature': 0.4,     # Less randomness
                'top_p': 0.85,
                'repetition_penalty': 2.0,
                'no_repeat_ngram_size': 3,
                'bad_words_ids': self.bad_words_ids,
                'eos_token_id': tokenizer.eos_token_id
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load explanation model: {e}", exc_info=True)
            self.explainer = None

    def generate_explanation(self, user_data, product_data, recommendation_score):
        """Generate recommendation explanation"""
        if not self.explainer:
            return "Explanation not available"
        
        # Validate required product data
        if not all(key in product_data for key in [ 'category']):
            return self._generate_fallback_explanation(product_data, recommendation_score)
        
        try:
            prompt = self._build_prompt(user_data, product_data, recommendation_score)
            output = self.explainer(prompt, **self.generation_config)[0]['generated_text']
            explanation = self._clean_explanation(output, prompt)
            
            if not self._is_valid_explanation(explanation):
                return self._generate_fallback_explanation(product_data, recommendation_score)
                
            return explanation
            
        except Exception as e:
            self.logger.error(f"Explanation generation error: {e}", exc_info=True)
            return self._generate_fallback_explanation(product_data, recommendation_score)

    def _build_prompt(self, user_data, product_data, score):
        """Build optimized English prompt for recommendation explanation"""
        return f"""
        You are an expert product recommendation assistant. Write a concise, natural explanation (2-3 sentences max) focusing on:
        1. Why this product suits the user's preferences
        2. Key product benefits
        3. Why it's recommended based on its rating

        Rules:
        - Never include contact information or links
        - No marketing phrases or promotions
        - Use natural, conversational language
        - Focus on value to the user

        User Profile:
        - Interests: {user_data.get('preferences', 'Not specified')}

        Product Info:
        - Name: {product_data.get('title', 'This product')}
        - Category: {product_data.get('category', 'this category')}
        - brand: {product_data.get('brand', 'this brand')}
        - Rating: {score:.2f}/5.0

        Example: "We recommend this {product_data.get('category', 'product')} because it matches your interests in {product_data.get('category', 'this category')} and offers excellent quality from {product_data.get('brand', 'a trusted brand')}. Its {score:.2f} rating shows consistent customer satisfaction."

        Start your explanation with: "We recommend this product because..."
        """

    def _clean_explanation(self, explanation, prompt):
        """Advanced output cleaning"""
        # Remove prompt if present
        explanation = explanation.replace(prompt, '').strip()
        
        # Remove email addresses
        explanation = re.sub(r'\S+@\S+', '', explanation)
        
        # Filter sentences by quality
        valid_sentences = []
        seen = set()
        
        for sentence in explanation.split('.'):
            clean_sentence = sentence.strip()
            if (clean_sentence 
                and len(clean_sentence.split()) >= 4  # Minimum word count
                and clean_sentence.lower() not in seen
                and not any(bad in clean_sentence.lower() for bad in self.bad_words)
            ):
                seen.add(clean_sentence.lower())
                valid_sentences.append(clean_sentence)
        
        # Return best 2-3 sentences
        if valid_sentences:
            return '. '.join(valid_sentences[:3]) + '.'
        return "Unable to generate suitable explanation"

    def _is_valid_explanation(self, text):
        """Quality check for generated explanations"""
        if not text or len(text.split()) < 8:  # Minimum length
            return False
        if any(bad in text.lower() for bad in self.bad_words):
            return False
        if text.count('.') < 1:  # Should contain complete sentences
            return False
        return True

    def _generate_fallback_explanation(self, product_data, score):
        """Fallback explanation when generation fails"""
        return f"We recommend {product_data.get('title', 'this product')} as it has a {score:.2f}/5.0 rating in the {product_data.get('category', 'relevant category')}."