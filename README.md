The script that implements logistic regression classification 
on bag of words features based on SIFT algorithm.

## Usage

1. Install `scipy, opencv, scikit-learn`:

       pip install scipy opnecv-python opencv-contrib-python scikit-learn
    
2. Run `bow_logistic_regresion.py` with an absolute or a relative path to images
that are evaluated. Images need to be in directories where directory present a
the class which image belongs to:
    
       python bow_logistic_regression.py <path to images> 