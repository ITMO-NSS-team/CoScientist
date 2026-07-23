I want to change the direction we write AAAI27 paper with.
Before we wrote CoScientist/alembic/docs/aaai27-plan.md and conducted experiments with quite a pessimstic outlook.
Now the idea (written in russian, coninue reasoning and all futher documents in english) is as follows - and goes more towards showing alembic as a very purpose driven part of an AI4Science ecosystem.
It resembes abstract and intro - we need abstract soon. Text is of course rough and need to be more appropriately worded, expanded and specified

Title: Towards generalisation of AI4Science systems 

Проблематика современных и AI4Science систем в том что они если и выполняют численные эксперименты то либо кодер агентом, либо фиксированными тулами
В крайнем случае набор готовых MCP (1)
Однако это приводят к тому что кодер генерация дорогая и стохастическая, результаты перевыполнения одних и тех же действий могут отличаться форматом и значением (не относится к изначально шумным генерациям и тд - там это в порядке нормы)

Научных MCP серверов слишком мало для покрытия больших исследований
Таким образом в текущих и AI4Science часть экспериментов довольно ограничена. (2)

Однако если бы существовала расширяемая экосистема научного MCP инструментария, то эксперименты проводимые в рамках AI4Science могли бы больше опираться на код предметных специалистов и менее зависимы от стохастических кодер агентов.
Поэтому создавая general AI4Science систему мы реализовываем в ней модуль экспериментов которые в процессе своей работы создает экосистему научного инструментария 
Он позволяет решать предметные задачи через код учёных и создавать каталог воспроизводимого научного кода.
Это должно позволить обеспечить воспроизводимость исследований для дальнейших Экспериментов.
Тезисы: 
1) мы стараемся создать универсальную AI4Science систему 
2) мы реализуем в ней систему создания динамичного расширяемого инструментального воспроизводимого научного кода, 
3) благодаря этому мы избавляемся от текущих проблем с экспериментов с: ограниченные инструменты, стохастика, стоимость
4) MCP которые мы создаем составляют экосистему открытую для научного кода которую можно переиспользовать

Basically the idea is that we tell about coscientist's planner, action graph and executor modules (WIP) that will actually call alembic, get it to set up MCP server and then call it to perform a scientific task. CoScientist's server ecosystem will enrich ovber time, as we convert more repoos available for use.

1. Do some additional background research on the idea as a whole and review papers that could support or falsify claims (1/2) + how do the findings we get before affect this idea.

2. Also search for benchmarks with scientific domain tasks that would have some repo that contains scientific code, task and some sort of gold validation In ToolMaker /home/stas/Documents/GitHub/ToolMaker there are tests which validate generated tool's output either on stric value match or expected output format
In ToolTosella's bench /home/stas/Documents/GitHub/ToolRosella-main/2603.09290v5.pdf they have a good set of scientific repos and tasks https://huggingface.co/datasets/ArthurY/ToolRosella/viewer/downstream,  but don't have gold AFAIK 
We kind of rely less on the presence of strict checks for MCP apart from out inner gates, but i'd expect a perfect scientific bench task to have a well-defined gold (even better if the same tool would need to be used on different input data, again as in ToolMaker's tests)

3. Formulate this into intro and abstract, initiate a latex project on AAAI27 template in CoScientist/alembic/docs/paper-aaai27

4. Outline the paper structure (will be of course edited, rough for now) as to be approprieate for this conference.