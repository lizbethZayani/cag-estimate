# Conversation page. A local ChatSession mirrors the session held by the API;
# the API owns the memory (history window, metadata), Rails keeps the remote id,
# the latest metadata snapshot and the estimations produced in the conversation.
class ChatSessionsController < ApplicationController
  include ConversationPage

  def new
    chat_session = current_chat_session || create_conversation
    redirect_to chat_session_path(chat_session)
  rescue EstimatorAi::Client::ServerError, Faraday::Error => e
    log_api_failure(e)
    render :unavailable, status: :service_unavailable
  end

  def show
    chat_session = ChatSession.find_by(id: params[:id])
    return redirect_to(new_chat_session_path) unless chat_session

    session[SESSION_KEY] = chat_session.id
    load_conversation(chat_session, max_turns: snapshot_max_turns(chat_session))
  rescue EstimatorAi::Client::SessionNotFound
    expire_conversation(chat_session)
  end

  # "New conversation": forget the remote session (best effort) and the mirror.
  # Estimations stay in the history.
  def destroy
    chat_session = ChatSession.find_by(id: params[:id])
    if chat_session
      delete_remote(chat_session)
      chat_session.destroy!
      session.delete(SESSION_KEY) if session[SESSION_KEY] == chat_session.id
    end
    redirect_to new_chat_session_path, status: :see_other
  end

  private

  def create_conversation
    remote_id = EstimatorAi::Client.new.create_session
    chat_session = ChatSession.create!(remote_session_id: remote_id)
    session[SESSION_KEY] = chat_session.id
    chat_session
  end

  # Only SessionNotFound propagates (the conversation is gone); any other
  # failure just leaves the "max turns" figure out of the panel.
  def snapshot_max_turns(chat_session)
    snapshot = EstimatorAi::Client.new.get_session(chat_session.remote_session_id)
    snapshot["max_turns"] if snapshot.is_a?(Hash)
  rescue EstimatorAi::Client::ServerError, Faraday::Error => e
    log_api_failure(e)
    nil
  end

  def delete_remote(chat_session)
    EstimatorAi::Client.new.delete_session(chat_session.remote_session_id)
  rescue EstimatorAi::Client::ServerError, Faraday::Error => e
    log_api_failure(e)
  end
end
