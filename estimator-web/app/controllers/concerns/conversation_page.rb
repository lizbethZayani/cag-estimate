# Shared plumbing for the conversation page (ChatSessionsController renders it,
# MessagesController re-renders it on errors).
module ConversationPage
  extend ActiveSupport::Concern

  SESSION_KEY = :current_chat_session_id
  UNAVAILABLE_MESSAGE = EstimationsController::UNAVAILABLE_MESSAGE
  EXPIRED_MESSAGE = "The conversation expired. A new one was started.".freeze

  private

  def load_conversation(chat_session, request: SessionEstimationRequest.new, max_turns: nil)
    @chat_session = chat_session
    @request      = request
    @max_turns    = max_turns
    @estimations  = chat_session.estimations.order(:created_at, :id)
  end

  def current_chat_session
    id = session[SESSION_KEY]
    id && ChatSession.find_by(id: id)
  end

  # The remote conversation is gone (API restart or expiry): drop the local
  # mirror and start over.
  def expire_conversation(chat_session)
    chat_session&.destroy
    session.delete(SESSION_KEY)
    flash[:alert] = EXPIRED_MESSAGE
    redirect_to new_chat_session_path
  end

  def log_api_failure(error)
    Rails.logger.error("Estimator API failure: #{error.class}: #{error.message}")
  end
end
