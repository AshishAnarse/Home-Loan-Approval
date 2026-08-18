# Home-Loan-Approval
ML project to automate Home Loan Approval from the data provided by applicants in application form.

## a.) The objective of this project is to develope and compare multiple machine-learning classification models for predicting whether a loan application will be approved or rejected.
## b.) The project uses a loan saction classification dataset containing information about loan applicants and their applications.
    ### Dataset size:
          instances: 614
          columns: 13
          Target Variable: Loan_Status
          Classification type: Binary
    ### Target Class distribution:
          Approved: 422 (68.73%)
          Rejected: 192 (31.27%)
    The dataset is therefore moderately imbalanced towards the approved class
## c.) Github repository link: https://github.com/AshishAnarse/Home-Loan-Approval
## d.) Models used:
    ### 1. Logistic Regression: 
            Accuracy: 0.8618    AUC: 0.8746    Precision: 0.8400    Recall: 0.9882    F1: 0.9081    MCC: 0.6721
    ### 2. Decision Tree:
            Accuracy: 0.7480    AUC: 0.7376    Precision: 0.8553    Recall: 0.7647    F1: 0.8075    MCC: o.4519
    ### 3. kNN:
            Accuracy: 0.8943    AUC: 0.8870    Precision: 0.9000    Recall: 0.9529    F1: 0.9257    MCC: 0.7468
    ### 4. Naive Bayes:
            Accuracy: 0.8455    AUC: 0.8372    Precision: 0.8367    Recall: 0.9647    F1: 0.8962    MCC: 0.6242
    ### 5. Random Forest:
            Accuracy: 0.8699    AUC: 0.8627    Precision: 0.8710    Recall: 0.9529    F1: 0.9101    MCC: 0.6856

    ### Observations:
        Logistic Regression: It provided strong performance and was particularly effective in minimizing false negatives for the approved class.
        Decision Tree: The lower test performance can be associated with tendency of decision tree to fit patterns and generalizing less efectively to unseen observations.
        kNN: The kNN results suggest that applicants with similar standardized demographics, financial, credit and loan characteristics tend to have similar loan approval outcomes.
        
