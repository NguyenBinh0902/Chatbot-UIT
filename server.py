from fastapi import FastAPI
from main import chain
from pydantic import BaseModel


app = FastAPI(
    title="UIT GPT",
)


class Message(BaseModel):
    msg: str


@app.post("/")
async def process_message(message: Message):
    result = chain.invoke(message.msg)
    return {"msg": result}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
