# POST /chat_sessions/:chat_session_id/messages: one conversation turn.
class MessagesController < ApplicationController
  include ConversationPage

  before_action :set_chat_session

  def create
    @request = SessionEstimationRequest.new(message_params)
    return render_conversation(:unprocessable_entity) unless @request.valid?

    payload = EstimatorAi::Client.new.estimate_in_session(
      @chat_session.remote_session_id, @request, attachments: attachments
    )
    record_turn(payload)
    redirect_to chat_session_path(@chat_session)
  rescue EstimatorAi::Client::GuardrailViolation, EstimatorAi::Client::InvalidRequest => e
    render_conversation(:unprocessable_entity, alert: e.message, kind: "rejected")
  rescue EstimatorAi::Client::SessionNotFound
    expire_conversation(@chat_session)
  rescue EstimatorAi::Client::ServerError, Faraday::Error => e
    log_api_failure(e)
    render_conversation(:service_unavailable, alert: UNAVAILABLE_MESSAGE, kind: "unavailable")
  end

  private

  def set_chat_session
    @chat_session = ChatSession.find_by(id: params[:chat_session_id])
    redirect_to new_chat_session_path unless @chat_session
  end

  def message_params
    params.fetch(:session_estimation_request, {}).permit(:transcript, :hourly_rate, :tier)
  end

  def attachments
    Array(params[:attachments]).select { |file| file.respond_to?(:original_filename) }
  end

  def record_turn(payload)
    ChatSession.transaction do
      @chat_session.estimations.create!(
        transcription:    @request.transcript,
        hourly_rate:      @request.hourly_rate,
        response_payload: payload,
        prompt_version:   payload["prompt_version"],
        cached:           false
      )
      @chat_session.update!(
        latest_metadata: payload["project_metadata"] || {},
        turn_count:      payload["turns"] || @chat_session.turn_count
      )
    end
  end

  def render_conversation(status, alert: nil, kind: nil)
    if alert
      flash.now[:alert] = alert
      flash.now[:alert_kind] = kind
    end
    load_conversation(@chat_session, request: @request)
    render "chat_sessions/show", status: status
  end
end
