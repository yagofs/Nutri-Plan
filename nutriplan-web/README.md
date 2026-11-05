# NutriPlan no RuFaS

Interface local instalada exclusivamente nesta pasta. Inicie com `iniciar_windows.bat` ou, nesta pasta, com `..\venv\Scripts\python.exe server.py`. Abra `http://127.0.0.1:8765`. O iniciador usa o ambiente virtual do RuFaS e não instala pacotes.

## Modos de cálculo

- **Configuração e Resultados:** modelo linear didático do ZIP, executado por SciPy/HiGHS. As exigências são fornecidas pelo usuário. Peso e leite são identificação; não geram exigências. Os alimentos e preços iniciais são exemplos históricos ou hipotéticos: substitua e documente as fontes. O resultado é identificado como didático.
- **Motor RuFaS:** lê um JSON completo, importa `Feed`, `NutritionRequirements` e `RationOptimizer` diretamente da instalação pai e chama `attempt_optimization`. É necessário fornecer os campos reais de ambos os objetos, aminoácidos, peso, DMI e proteína metabolizável. O modelo JSON baixável é deliberadamente incompleto, para não inventar dados científicos. Somente NRC com `Feed` é aceito nesta interface; NASEM requer `NASEMFeed` e entradas adicionais. A moeda é a definida nos preços fornecidos pelo usuário; `purchase_cost` deve estar por kg de MS. O servidor apresenta as folgas das restrições nativas.

O fluxo simplificado não fornece os dados científicos exigidos pelo RuFaS e não é convertido automaticamente para ele. Uma aplicação acadêmica concluída exige dados de composição e exigências validados, com fonte e unidade para cada campo, e cenários nativos revisados por especialista.

## API local

- `GET /api/health`: disponibilidade do servidor e presença do otimizador.
- `POST /api/optimize`: modelo didático.
- `POST /api/rufas-native`: otimizador RuFaS com JSON completo.

O servidor escuta apenas em `127.0.0.1`, sem modo debug. Nenhum módulo original do RuFaS foi alterado. Não use resultados ilustrativos para formulação real de dietas.
