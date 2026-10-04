class EstimationsController < ApplicationController
  UNAVAILABLE_MESSAGE = "AI service unavailable. Please try again in a moment.".freeze

  def index
    @estimations = Estimation.order(created_at: :desc).limit(20)
  end

  def new
    @request = EstimationRequest.new
  end

  def create
    @request = EstimationRequest.new(estimation_request_params)

    unless @request.valid?
      render :new, status: :unprocessable_entity
      return
    end

    payload = EstimatorAi::Client.new.estimate(@request)

    @estimation = Estimation.create!(
      transcription:    @request.transcription,
      hourly_rate:      @request.hourly_rate,
      response_payload: payload,
      prompt_version:   payload["prompt_version"],
      cached:           payload["cached"] || false
    )

    redirect_to estimation_path(@estimation)
  rescue EstimatorAi::Client::GuardrailViolation, EstimatorAi::Client::InvalidRequest => e
    flash.now[:alert] = e.message
    render :new, status: :unprocessable_entity
  rescue EstimatorAi::Client::ServerError, Faraday::ConnectionFailed, Faraday::TimeoutError => e
    Rails.logger.error("Estimator API failure: #{e.class}: #{e.message}")
    flash.now[:alert] = UNAVAILABLE_MESSAGE
    render :new, status: :service_unavailable
  end

  def show
    @estimation = Estimation.find(params[:id])
    @response   = @estimation.to_response
  end

  private

  def estimation_request_params
    params.require(:estimation_request).permit(:transcription, :hourly_rate)
  end
end
