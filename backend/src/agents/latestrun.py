def run(self, question, session_id=None):

        logger.info("⚙️ EngineeringAgent running")

        # ---------------- QUERY REWRITE ----------------
        if Config.ZERO_LLM_MODE:
            improved_query = question
        else:
            try:
                improved_query = self.rewriter.rewrite(question)
            except Exception as e:
                logger.warning(f"Rewrite failed: {e}")
                improved_query = question

        improved_query = f"{improved_query} {question}"

        logger.info(f"Query: {improved_query}")

        # =====================================================
        # 🔥 MULTI QUERY GENERATION (NEW)
        # =====================================================
        queries = self._generate_multi_queries(improved_query)

        logger.info(f"Generated Queries: {queries}")

        # =====================================================
        # 🔥 MULTI RETRIEVAL
        # =====================================================
        all_docs = []

        for q in queries:
            try:
                docs = self.retriever.retrieve(q)
                logger.info(f"Retrieved {len(docs)} docs for query: {q}")
                all_docs.extend(docs)
            except Exception as e:
                logger.warning(f"Retrieval failed for query [{q}]: {e}")

        if not all_docs:
            return {
                "answer": "No relevant documents found.",
                "docs": [],
                "confidence": 0.0,
                "rag_conf": 0.0,
                "graph_conf": 0.0
            }

        # =====================================================
        # 🔥 DEDUPLICATION (CRITICAL)
        # =====================================================
        unique_docs = {}
        for doc in all_docs:
            key = doc.page_content[:200]  # lightweight unique key
            if key not in unique_docs:
                unique_docs[key] = doc

        merged_docs = list(unique_docs.values())

        logger.info(f"Total docs after dedup: {len(merged_docs)}")

        # =====================================================
        # 🔥 GLOBAL RERANK (CrossEncoder)
        # =====================================================
        top_docs = self.reranker.rerank(improved_query, merged_docs)

        logger.info(f"Top docs after reranker.rerank: {len(top_docs)}")

        # =====================================================
        # 🔥 MEMORY + CUSTOM RANK
        # =====================================================
        memory_context = self._get_memory_context(session_id)
        top_docs = self._rank_context(question, top_docs, memory_context)

        logger.info(f"Total top docs after rank context and memory: {len(top_docs)}")

        if not top_docs:
            return {
                "answer": "No relevant documents found.",
                "docs": [],
                "confidence": 0.0,
                "rag_conf": 0.0,
                "graph_conf": 0.0
            }

        # =====================================================
        # 🔥 DEBUG LOG
        # =====================================================
        logger.info("======== CONTEXT DEBUG ========")
        for i, d in enumerate(top_docs[:3]):
            logger.info(f"DOC {i+1} preview: {d.page_content[:200]}")
        logger.info("================================")

        # =====================================================
        # 🔥 CONFIDENCE
        # =====================================================
        rag_conf = self._calculate_confidence(improved_query, top_docs)

        # =====================================================
        # 🔥 GRAPH
        # =====================================================
        graph_text = ""
        graph_conf = 0.0

        if self.graph_agent:
            try:
                g = self.graph_agent.run(question, session_id)
                if isinstance(g, dict):
                    graph_text = g.get("answer", "")
                    graph_conf = g.get("confidence", 0.0)
                else:
                    graph_text = str(g)
                    graph_conf = 0.5
            except Exception as e:
                logger.warning(f"GraphAgent failed: {e}")

        logger.info(f"RAG: {rag_conf} | GRAPH: {graph_conf}")

        # =====================================================
        # 🔥 CONTEXT BUILD
        # =====================================================
        context = []

        for i, doc in enumerate(top_docs[:5]):

            page = doc.metadata.get("page_label") or doc.metadata.get("page") or "Unknown"

            context.append(f"""
                --- DOCUMENT {i+1} ---
                Source: {os.path.basename(doc.metadata.get("source",""))}
                Page: {page}

                {doc.page_content}
                """)

        context_text = "\n\n".join(context)

            # =====================================================
            # 🔥 STRONG PROMPT
            # =====================================================
        prompt = f"""
            You are a senior civil engineer.

            STRICT RULES:
            1. Use ONLY the provided documents
            2. DO NOT ignore any relevant section
            3. DO NOT invent information
            4. Every statement MUST be supported by text from the documents
            5. ALWAYS cite like (Document 1, Page X)
            6. If unsure → say "Not found in documents"

            Documents:
            {context_text}

            Graph:
            {graph_text}

            Question:
            {question}

            Return:
            - Final Answer (with citations)
            - Technical Explanation
            - Confidence (0 to 1)
            - Supporting Evidence (exact lines)
            """

        logger.info(f"\nPROMPT:\n{prompt[:1500]}")

        # =====================================================
        # 🔥 LLM
        # =====================================================
        try:
            response = self.llm.invoke(prompt).content

            '''if rag_conf < 0.7:
                is_valid, reason = self._strict_verify(response, top_docs)
            else:
                is_valid, reason = self._verify_answer(response, top_docs)

            if not is_valid:
                logger.warning("⚠️ Answer not grounded")
            
                return {
                    "answer": response,
                    "docs": top_docs[:5],
                    "confidence": round(max(rag_conf, graph_conf), 2),
                    "rag_conf": rag_conf,
                    "graph_conf": graph_conf
                }'''

        except Exception as e:
            logger.error(f"LLM failed: {e}")
            return {
                "answer": top_docs[0].page_content,
                "docs": top_docs[:5],
                "confidence": rag_conf,
                "rag_conf": rag_conf,
                "graph_conf": graph_conf
            }

        logger.info(f"\nLLM RESPONSE:\n{response}")

        return {
            "answer": response,
            "docs": top_docs[:5],
            "confidence": round(max(rag_conf, graph_conf), 2),
            "rag_conf": rag_conf,
            "graph_conf": graph_conf
        }
