import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { ConversationalRetrievalQAChain } from "langchain/chains";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { BufferMemory } from "langchain/memory";
import { TextLoader } from "langchain/document_loaders/fs/text";

import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import * as fs from "fs";

import dotenv from 'dotenv';

const run = async () => {
  dotenv.config();
  const key = process.env.APIKEY;
  
  
  if (process.argv.length < 3) {
    console.log("Please enter a prompt");
    process.exit(1);
  }
  const model = new ChatOpenAI({
    openAIApiKey: key,
  });

  const text = fs.readFileSync("data/data.txt", "utf8");
  const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
  const docs = await textSplitter.createDocuments([text]);
  
  // Load the docs into the vector store
  const vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings({
    openAIApiKey: key,
  }));
  

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorStore.asRetriever(),
    {
      memory: new BufferMemory({
        memoryKey: "chat_history", // Must be set to "chat_history"
      }),
    }
  );
  var question = process.argv[2]; 
  const res = await chain.call({ question: question });
  console.log(res.text)

};

run();