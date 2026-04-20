import uvicorn

if __name__ == "__main__":
    # Run the InfraAi server directly using uvicorn
    uvicorn.run("infra_ai.api.main:app", host="127.0.0.1", port=8081, reload=True)