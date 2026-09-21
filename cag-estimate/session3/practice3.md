practice 3

mvp: wrappers, treaming and trazability.
- abtract layer: LLM calls to provraider wrapper. change of the provider no chaniging lines of code!
- fallback automatic: if any of the providers fails then call the another enabled provider, then the service is always live!
- Smart catching: the repetitive transcriptions most not call the model again then the service is saving tokens and the reponse is instantly.
- Streaming: servers center to see the calls chat conversation token in real time then we are aware if the chat is really work.
- Trazability: each call is register with each model, cost, its catching or not.

goal of this practice: 
Build an UI chat web with Streamlit. We can chat with the transcriptions of the estimations projects and then the estimation than the LLM is creating is visibile in real time, then each token created is showing up at the moment.

format: stramlit_app.py in the root of the project, run it with streamlit run streamlit_app.py

Tasks

1.- Chat: with Streamlit as interface of chat st.chat_message, st.chat_input, writing the conversation, or paste etc. Then we sent the input text to the LLM and then showing up the result of the assistan.
rules: the history of the conversations most tbe visible during the session with st.session_state.
the systmem prompt most be the same than we use with the CAG endpopint

2.-Streaming: we want to do it very friendly and real time as we have a simple conversation with a friend on whatsapp app. then when the assitance llm is creating the tokens then we might show up immediatly instead of wait for all the conversation token done. we most see he estimation writing on real time....
rules: use st.write.stream

3.- CAG context at the interface
For visibility and obersbaility about what we are using, create a panel in the lateral side of the ui st.sidebar showing up:
the system prompt activo as reading, the static context inyected ( the estimation examples than support el CAG), basics metrics of the last llm calls where show model using, inputs, outputs tokens, times responses, total cost in dlls.

4.- Long conversations:
we most set max_tokens=2000, this is enought for get done the assitant. We most dectect  if the finish_reason="length", instead of stop, the call was cut because this max tokens in this case we have to request a call llm to conitinue with this otherwise the user wont see nothing, pretty bad. so we have to handel it.

verifations:
1.- streamlit run streamlit_app.py open the ui with the chat in the web
2.- type, paste tc the conversation and the get the estimations back
3.- the chat with the conversations persist in the screen and then we can do chats followings
4.- the assitant answear showing up in streaming all the tokens done at the real time

