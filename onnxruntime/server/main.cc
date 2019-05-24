// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "environment.h"
#include "http_server.h"
#include "predict_request_handler.h"
#include "server_configuration.h"

namespace beast = boost::beast;
namespace http = beast::http;
namespace server = onnxruntime::server;

int main(int argc, char* argv[]) {
  server::ServerConfiguration config{};
  auto res = config.ParseInput(argc, argv);

  if (res == server::Result::ExitSuccess) {
    exit(EXIT_SUCCESS);
  } else if (res == server::Result::ExitFailure) {
    exit(EXIT_FAILURE);
  }

  const auto env = std::make_shared<server::ServerEnvironment>(config.logging_level);
  auto logger = env->GetAppLogger();
  LOGS(logger, VERBOSE) << "Logging manager initialized.";
  LOGS(logger, INFO) << "Model path: " << config.model_path;

  auto status = env->InitializeModel(config.model_path);
  if (!status.IsOK()) {
    LOGS(logger, FATAL) << "Initialize Model Failed: " << status.Code() << " ---- Error: [" << status.ErrorMessage() << "]";
    exit(EXIT_FAILURE);
  } else {
    LOGS(logger, VERBOSE) << "Initialize Model Successfully!";
  }

  status = env->GetSession()->Initialize();
  if (!status.IsOK()) {
    LOGS(logger, FATAL) << "Session Initialization Failed:" << status.Code() << " ---- Error: [" << status.ErrorMessage() << "]";
    exit(EXIT_FAILURE);
  } else {
    LOGS(logger, VERBOSE) << "Initialize Session Successfully!";
  }

  auto const boost_address = boost::asio::ip::make_address(config.address);
  server::App app{};

  app.RegisterStartup(
      [&env](const auto& details) -> void {
        auto logger = env->GetAppLogger();
        LOGS(logger, INFO) << "Listening at: "
                           << "http://" << details.address << ":" << details.port;
      });

  app.RegisterError(
      [&env](auto& context) -> void {
        auto logger = env->GetLogger(context.request_id);
        LOGS(*logger, VERBOSE) << "Error code: " << context.error_code;
        LOGS(*logger, VERBOSE) << "Error message: " << context.error_message;

        context.response.result(context.error_code);
        context.response.insert("Content-Type", "application/json");
        context.response.insert("x-ms-request-id", context.request_id);
        if (!context.client_request_id.empty()) {
          context.response.insert("x-ms-client-request-id", (context).client_request_id);
        }
        context.response.body() = server::CreateJsonError(context.error_code, context.error_message);
      });

  app.RegisterPost(
      R"(/v1/models/([^/:]+)(?:/versions/(\d+))?:(classify|regress|predict))",
      [&env](const auto& name, const auto& version, const auto& action, auto& context) -> void {
        server::Predict(name, version, action, context, env);
      });

  app.Bind(boost_address, config.http_port)
      .NumThreads(config.num_http_threads)
      .Run();

  return EXIT_SUCCESS;
}
