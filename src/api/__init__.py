from .router import router


def create_app():
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(
        title="Standalone Bot API",
        description="Assist the user using the Knowledgebase",
        version="0.1.0",
        openapi_url="/api/openapi.json",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )

    app.max_request_size = 200 * 1024 * 1024

    # Set up CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Specify the correct frontend origin here
        allow_credentials=True,
        allow_methods=["*"],  # Allows all methods
        allow_headers=["*"],  # Allows all headers
    )

    app.include_router(router, tags=["Standalone Bot"], prefix="/api")

    return app