import { OpenAI } from 'langchain/llms/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { ConversationalRetrievalQAChain } from 'langchain/chains';

const CONDENSE_PROMPT = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`;

const QA_PROMPT = `
Eres un útil asistente de inteligencia artificial que debe proporcionar informacion sobre el estado de los colaboradores al area de recursos humanos.
Utiliza los siguientes fragmentos de contexto para responder.
Responde en el mismo idioma de la pregunta.
Si no sabes la respuesta, simplemente di que no lo sabes. NO intentes inventar una respuesta.
Si la pregunta no está relacionada con el contexto, responde educadamente que estás configurado para responder solo preguntas relacionadas con el contexto.

{context}

Question: {question}
Helpful answer in markdown:`;

export const makeChain = (vectorstore: PineconeStore) => {
  const model = new OpenAI({
    temperature: 0.8, // increase temepreature to get more creative answers
    //modelName: 'gpt-4', //change this to gpt-4 if you have access
    modelName: 'gpt-3.5-turbo', //change this to gpt-4 if you have access
  });

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorstore.asRetriever(),
    {
      qaTemplate: QA_PROMPT,
      questionGeneratorTemplate: CONDENSE_PROMPT,
      returnSourceDocuments: true, //The number of source documents returned is 4 by default
    },
  );
  return chain;
};
