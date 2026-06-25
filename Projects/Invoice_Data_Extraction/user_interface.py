import streamlit as st
from dotenv import load_dotenv
import invoice_util as iu
import os
load_dotenv()

def main():
    load_dotenv()

    st.set_page_config(page_title="Invoice Extraction Bot")
    st.title("Invoice Extraction Bot")
    st.subheader("I can help you in extraction invoice data")

    pdf = st.file_uploader("Uploader invoices here, onlu pdf files allowed", type=["pdf"], accept_multiple_files=True)

    submit = st.button("Extract data")

    if submit:
        with st.spinner('Wait for it'):
            df = iu.createdocs(pdf)
            st.write(df.head())

            data_as_csv = df.to_csv(index = False).encode("utf-8")
            st.download_button(
                "Download data as CSV", 
                data_as_csv, 
                "benchmark-tools.csv",
                "text/csv",
                key="download-tools-csv",
            )
        st.success("Hope I was able to save your time")



#Invoking main function
if __name__ == '__main__':
    main()

