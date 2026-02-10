# infiniteZoom RAG

infinite zoom knowledge graph powered by gemini. start from big topics, zoom in to discover subtopics, chapters, concepts, and facts — all generated on the fly. search navigates the graph live with a rendered video of the traversal.

**live:** https://gem3-zoom-51139704820.us-east5.run.app

## inspiration

we kept running into the same problem — you search for something, get a flat list of links, and lose all sense of how things connect. knowledge has structure. it has depth. we wanted to see that depth, literally zoom into it the way you'd zoom into a map. the idea was simple: what if you could start from "Science" and just keep zooming until you hit actual facts?

## what it does

infiniteZoom is a knowledge graph you navigate by zooming. it starts with big topics rendered as huge faded text on a canvas. zoom in and they expand — gemini generates subtopics, chapters, sections, concepts, all the way down to individual facts. there's a search that actually traverses the graph live — you watch it zoom through layers picking the most relevant path, then it gives you an answer with a rendered video of the traversal you can replay. the whole thing runs on a single python server with no frontend framework.

## how we built it

pure python backend with aiohttp serving a single HTML page. the canvas is just absolutely positioned spans with CSS transforms for zoom/pan — no canvas element, no WebGL, just DOM. gemini 3 flash generates children for each node when you zoom in far enough. search uses server-sent events to stream each step back to the browser while speculatively expanding nodes in parallel. the traversal video is rendered client-side to a hidden canvas with MediaRecorder. deployed on cloud run with bigquery for persistence.

## challenges we ran into

bigquery streaming buffer. you can't update rows you just inserted — there's a buffer delay. we burned time trying to track expanded state in BQ before just making it in-memory only. also batch expanding multiple nodes in one gemini call took some prompt engineering to get reliable JSON back. and making the zoom feel smooth with thousands of DOM nodes required aggressive LOD culling — only 3-4 layers visible at any time.

## accomplishments that we're proud of

the speculative expansion during search. while gemini picks which node to follow, we're already expanding ALL candidate nodes in parallel. so by the time it decides, the children are already there. zero wait. also the video replay — it renders the entire traversal to a canvas at 30fps and shows it next to the answer automatically. no libraries, no ffmpeg, just browser APIs.

## what we learned

you don't need react. you don't need a graph library. a few hundred lines of vanilla JS with CSS transforms can do infinite zoom with LOD rendering. also gemini 3 flash is fast enough to generate knowledge nodes on the fly without it feeling slow — the speculative parallelism helps a lot.

## what's next

grounded mode — we have the code for it already. ingest actual documents, build the tree from real content, and ground the leaf nodes in source text. so when you zoom all the way in you're reading actual excerpts from papers or docs, not generated text. also want to add collaborative graphs where multiple people explore the same canvas and see each other's paths.

## run locally

```bash
pip install -r requirements.txt
python main.py
```

## deploy

```bash
gcloud run deploy gem3-zoom --source . --project YOUR_PROJECT --region us-east5
```

## stack

- python / aiohttp
- gemini 3 flash (via vertex ai)
- bigquery (persistence)
- cloud run
- vanilla js / css transforms / MediaRecorder
