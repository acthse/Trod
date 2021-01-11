import asyncio

from helo import _db

db = _db.new("mysql://root:HELLOgjw123@127.0.0.1:3306/helo")


async def test():
    # async with db.connection().transaction():
    await db.execute(
        "INSERT INTO `HighScores` (`name`, `score`) VALUES ('jiawei', 100);"
    )
    await db.execute(
        "INSERT INTO `HighScores` (`name`, `score`) VALUES ('jiali', 200);"
    )
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx: ', len(await db.execute("select * from HighScores;")))


async def main():
    await db.connect()

    tasks = []
    for i in range(6):
        tasks.append(asyncio.create_task(test()))

    await asyncio.gather(*tasks)

    await db.close()

if __name__ == "__main__":
    asyncio.run(main())
