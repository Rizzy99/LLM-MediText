import streamlit as st
from PIL import Image

from importnb import Notebook

with Notebook():
    from LLM import full_rag_pipeline
 # Replace with the correct module name

# App Title
st.title("Medical Question-Answering System")
st.markdown("""
This application allows you to ask medical-related questions and get accurate, context-aware answers based on medical textbooks.
""")

# User Query Input
query = st.text_input("Enter your medical-related question:")

# Submit Button
if st.button("Submit"):
    if query.strip():
        # Process the query using the RAG pipeline
        with st.spinner("Retrieving and generating answer..."):
            response = full_rag_pipeline(query=query)
        
        # Display the generated answer
        st.subheader("Answer:")
        st.write(response["answer"])
        
        # Display contextual references
        st.subheader("Contextual References:")
        if response["evidence"]:
            for idx, evidence in enumerate(response["evidence"], start=1):
                st.write(f"**Reference {idx}:**")
                st.write(f"**ID:** {evidence['id']}")
                st.write(f"**Content:** {evidence['content']}")
        else:
            st.write("No contextual references available.")
    else:
        st.warning("Please enter a question.")

# Footer
st.markdown("""
---
*Powered by Flan-T5-small and Retrieval-Augmented Generation (RAG) pipeline*
""")
