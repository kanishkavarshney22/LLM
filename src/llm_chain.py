from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def run_llm_query(docs, user_query):
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = PromptTemplate.from_template("""
    You are a decision engine for insurance claims. Based on the following policy clauses and customer query, return a JSON with:

    - "decision": "approved" or "rejected"
    - "amount": if applicable
    - "justification": mention clause or rule used

    Context:
    {context}

    Query:
    {query}
    """)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({"context": context, "query": user_query})
