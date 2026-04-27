Ask something



first chat created



thinking



Orchestrator.answer get executed with prompt



go to Orchestrator.answer 



print question



in memory, add\_user\_message with question



call router class

call router.route (router.py under orchestration) method with question - This will decide which agent need to be called

Router.route -  check whether question is present in cache by converting it in hash

If already exist then it return it back to orchestrator.answer



if it does not exist then call \_fast\_classify which already exist in router.py same class

If \_fast\_classify check whether any particular word exist in question then base on it return intent mostly for graph. if nothing is present then it return rag

it also calculate the score and confidence

This \_fast\_classify return two values intent and confidence to route function



if confidence is more than 0.7 then it store the intent in cache and return intent back to orchestrator.answer function



If confidence is less than 0.7 then it call LLM fallback \_llm\_route with question



\_llm\_route execute the prompt with question. Prompt ask llm to return value as graph or rag or hybrid

If result is not in graph or rag or hybrid then it return wantning "Invalid LLM Output" and return intent as rag as default

If result from llm prompt contain some value then it return as intent

&#x20;

So route of router class check the intent whether it is graph or rag or hybrid by calling fast\_classify and then llm\_route methods and return intent back to orchestrator.answer method



orchestrator.answer method then check the value of intent

if it is hybrid then call hybrid\_answer which present in same class orchestrator with question

for graph - .graph\_agent.run with question

for rag - engineering\_agent.run

for wheather - weather\_agent.run

for tender - tender\_agent.run

for project - project\_agent.run

for else - engineering\_agent.run



if there is result (whether dict or tupple) by any particular agent then it store the values in variable



final result is written in following format



final\_result = {

&#x20;               "answer": text,

&#x20;               "docs": \[],

&#x20;               "rag\_conf": 0.0,

&#x20;               "graph\_conf": 0.0

&#x20;           }



This final result is return back to main method of streamlit from orchestrator.answer method



Then streamlit main call render\_response(result) which is present in same streamlit class



response check whether result is present and have some values

then it separate out result in following format



answer = result.get("answer", "")

&#x20;       docs = result.get("docs", \[])

&#x20;       rag\_conf = result.get("rag\_conf", None)

&#x20;       graph\_conf = result.get("graph\_conf", None)



Then same \_render\_response take answer and display under Answer on UI

Answer contain final answer and technical explaination



then it shows sources using docs variable

from docs it separate out source and page by looping through docs 

these sources and pages get displayed on screen



The it display rag confidence using rag\_conf variable on screen

then it display graph confidence using graph\_conf variable on screen



Then it display debug info 

under this it display complete result



Then it back to main method of streamlit\_app

it add answer in conversation by calling add\_message function



if it is new chat then it change the title by getting chat 40 characters and change the title in new chat



The main part of this application is engineering\_agent.run

suppose the intent is rag then it call engineeringagent.run

This method actually retrieve result from vectordb. Lets find the steps perform in this run method of engineering\_agent method



engineering\_agent.run take question and session id



If the Config.ZERO\_LLM\_MODE is true means do not want to use llm then improved\_query variable store question

else

&#x09;query is rewrite using self.rewriter.rewrite(question) - we will see rewriter.write in details later on



The improved\_query value is assigned again improved\_query (as may be rewriter.rewrite improve the query)



then improved\_query is passed as parameter to retriever.retreive to retrieve the docs which contain the result(answer, page etc) - we will see retriever.retreive in details later on



The improved\_query and docs(which are return by retriever.retrieve method) are passed as parameter to reranker.rerank  top\_docs = self.reranker.rerank(improved\_query, docs)

This method return top documents which store in top\_docs variable



Then it call memory\_context = self.\_get\_memory\_context(session\_id) to get memory context



then it call \_rank\_context with question and top\_docs which received from reranker.rerank



If not top\_docs then it return 

return {

&#x20;               "answer": "No relevant documents found.",

&#x20;               "docs": \[],

&#x20;               "confidence": 0.0,

&#x20;               "rag\_conf": 0.0,

&#x20;               "graph\_conf": 0.0

&#x20;           }



if top\_docs then

&#x09;it calculate the confidence of rag using \_calculate\_confidence

here now rag part is done then it move to graph\_agent:



it call graph\_agent.run with question and sesson\_id  and store in variable g



from g it extract graph\_text and graph\_conf ( graph text and graph confidence).



Then it build context using for loop (for i, doc in enumerate(top\_docs\[:5]):)

in this for loop it extract real page using extract\_real\_page method and store this context in context variable in following format

document id, source, page and page content



after completing of this for loop all the context present in context variable get concatenate in the variable context\_text

Then it generate prompt using 

Context\_text, graph\_text, question and return format



Then this prompt pass to llm 

then return output is return back to orchestrator.answer



Let see each function use in engineeringagent class (engineering\_agent)



1. rewriter.rewrite

&#x09;it take question and create prompt with question to rewrite the question and pass to llm

&#x09;llm response is sent back to engineeringagent.run function



2\. retriever.retriever (hybrid\_retriever.py)

&#x09;it take improved query and use vectorretriever to retrieve the result

&#x09;it first call vectore\_retriever.invoke with query self.vector\_retriever.invoke(query)

&#x09;Then it call keyword retriever self.keyword\_retriever.invoke(query) 

&#x09;Then combine the output 

&#x09;Then it loop all the documents (combined) to get unique docs

&#x09;Then it return unique documents values back to engineeringagent.run



3\. reranker.rerank(improved\_query, docs) reranker.py

&#x09;It take improved\_query and docs to rerank the documents
	first it split the query and store in query\_terms variable

&#x09;it take top 5 documents from docs array variable

&#x09;It loops all these documents

&#x09;	first it take all the page\_content of document and store in text variable

&#x09;	then it check split query terms present in text. if exist then it increases the score

&#x09;In scored\_docs variable it append score and that document from for loop

&#x09;then it sort all the dcouments present tin scored\_docs variable in descending order by score



&#x09;Last it return all the documents back to  engineeringagent.run



4\. get\_memory\_context(engineering\_agent.py) 

&#x09;The memory object is create in streamlitapp class in init\_system then this memory object is passed to the engineeringagent initialize as parameter. This memory object then use in engineeringAgent.run to get memory information

&#x09;get\_memory\_context check whether memory object and session\_id exist. Memory class is defined as conversationmemory in conversation\_memory.py file

&#x09;If exist then it get summary and recent history using session\_id. This get\_memory\_context returns summary and text (recent history). The return memory context is stored in memory\_context variable in run method of engineeringagent class

&#x09;

5.\_rank\_context(engineering\_agent.py)

&#x09;This method take three parameters  questions(enter by user), top\_docs (reranker.rerank - rerank documents) and memory context( return by get\_memory\_context



6\. self.\_calculate\_confidence

&#x09;it take improved query and top\_docs which return by \_rank\_context. 

&#x09;This function return rag confidence



7\. self.graph\_agent.run

&#x09;It return graph text(answer) and graph confidence

&#x09;



&#x09;

8\. \_extract\_real\_page (engineeringagent class)

&#x09;This function take two parameters doc.page\_content and doc.metadata.page\_label or doc.metadata.page

&#x09;it loops all the docs from top\_docs array

&#x09;In this loop, it keep adding the page information in context variable

&#x09;

&#x09;































































































