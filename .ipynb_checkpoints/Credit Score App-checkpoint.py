import numpy as np
import joblib
import streamlit as st

# loading the saved model
scaler = joblib.load('scaler.pkl')
loaded_model = joblib.load('model.pkl')


# creating a function for Prediction

def diabetes_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if prediction[0] == 0:
      return "Credit Score is Good"
    elif prediction[0] == 1:
      return "Credit Score is Poor"
    else:
      return "Credit Score is Standard"
  
    
  
def main():
    
    
    # giving a title
    st.title('Credit Score Prediction')
    
    
    # getting the input data from the user
    
    
    Annual_Income = st.text_input('Person Annual Income')
    Monthly_Inhand_Salary = st.text_input('Monthly in Hand Salary')
    Interest_Rate = st.text_input('Interest Rate')
    Num_of_Loan = st.text_input('Number of Loans')
    Delay_from_due_date = st.text_input('Number of Delayed Days')
    Num_of_Delayed_Payment = st.text_input('Number of Delayed Payments')
    Credit_Mix = st.text_input('Credit Mix')
    Outstanding_Debt = st.text_input('Outstanding Debt')
    Credit_Utilization_Ratio = st.text_input('Credit Card Utilization Ratio')
    Payment_of_Min_Amount = st.text_input('Payment of Minimum Amount')
    Total_EMI_per_month = st.text_input('Equated Monthly Installment')
    Amount_invested_monthly = st.text_input('Amount Invested Monthly')
    Monthly_Balance = st.text_input('Monthly Balance')
    Credit_History_Age_In_Years = st.text_input('Credit History in Years')
    StudentLoan = st.text_input('1 if get the loan otherwise 0')
    MortgageLoan = st.text_input('1 if get the loan otherwise 0')
    PersonalLoan = st.text_input('1 if get the loan otherwise 0')
    DebtConsolidationLoan = st.text_input('1 if get the loan otherwise 0')
    Credit_BuilderLoan = st.text_input('1 if get the loan otherwise 0')
    HomeEquityLoan = st.text_input('1 if get the loan otherwise 0')
    AutoLoan = st.text_input('1 if get the loan otherwise 0')
    PaydayLoan = st.text_input('1 if get the loan otherwise 0')
    NotSpecified_Loan = st.text_input('1 if get the loan otherwise 0')
    
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Annual_Income, Monthly_Inhand_Salary, Interest_Rate, Num_of_Loan, Delay_from_due_date, Num_of_Delayed_Payment, Credit_Mix, Outstanding_Debt, 
                                         Credit_Utilization_Ratio, Payment_of_Min_Amount, Total_EMI_per_month, Amount_invested_monthly, Monthly_Balance, Credit_History_Age_In_Years, StudentLoan,
                                         MortgageLoan, PersonalLoan, DebtConsolidationLoan, Credit_BuilderLoan, HomeEquityLoan, AutoLoan, PaydayLoan, NotSpecified_Loan])
        
        
    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()