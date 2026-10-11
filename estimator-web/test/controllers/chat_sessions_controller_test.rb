require "test_helper"

class ChatSessionsControllerTest < ActionDispatch::IntegrationTest
  setup do
    @base = "#{Rails.application.config.estimator_ai.base_url}/api/v1/sessions"
  end

  def stub_create(session_id: "sess-1", status: 201)
    stub_request(:post, @base).to_return(
      status: status, body: { session_id: session_id }.to_json,
      headers: { "Content-Type" => "application/json" }
    )
  end

  def stub_snapshot(session_id = "sess-1", status: 200, body: { session_id: session_id, max_turns: 6 })
    stub_request(:get, "#{@base}/#{session_id}").to_return(
      status: status, body: body.to_json, headers: { "Content-Type" => "application/json" }
    )
  end

  def create_turn(chat_session, text:, payload: session_estimation_payload)
    chat_session.estimations.create!(
      transcription: text, hourly_rate: 40, response_payload: payload, prompt_version: "v2"
    )
  end

  test "root is the conversation entry point" do
    assert_recognizes({ controller: "chat_sessions", action: "new" }, "/")
  end

  test "GET new creates the remote and local session lazily and redirects to show" do
    create = stub_create
    assert_difference "ChatSession.count", 1 do
      get new_chat_session_path
    end
    chat_session = ChatSession.order(:id).last
    assert_equal "sess-1", chat_session.remote_session_id
    assert_redirected_to chat_session_path(chat_session)
    assert_requested create, times: 1
  end

  test "GET new again reuses the current conversation" do
    create = stub_create
    get new_chat_session_path
    assert_no_difference "ChatSession.count" do
      get new_chat_session_path
    end
    assert_requested create, times: 1
    assert_redirected_to chat_session_path(ChatSession.order(:id).last)
  end

  test "GET new shows a friendly 503 page when the API is down" do
    stub_request(:post, @base).to_return(status: 502, body: { detail: "SECRET upstream" }.to_json,
                                         headers: { "Content-Type" => "application/json" })
    assert_no_difference "ChatSession.count" do
      get new_chat_session_path
    end
    assert_response :service_unavailable
    assert_match "AI service unavailable", response.body
    assert_no_match "SECRET", response.body
  end

  test "GET new shows a friendly 503 page on timeout" do
    stub_request(:post, @base).to_timeout
    get new_chat_session_path
    assert_response :service_unavailable
    assert_match "AI service unavailable", response.body
  end

  test "GET show renders an empty conversation with the message form and metadata panel" do
    chat_session = ChatSession.create!(remote_session_id: "sess-1")
    stub_snapshot
    get chat_session_path(chat_session)

    assert_response :success
    assert_select "[data-empty-state]", text: /Send your first message/
    assert_select "form[action='#{chat_session_messages_path(chat_session)}'][enctype='multipart/form-data'][data-controller~='form-loading']"
    assert_select "textarea[name='session_estimation_request[transcript]']"
    assert_select "input[type=file][name='attachments[]'][multiple][accept='.pdf,.docx,.txt,.md,.csv,.json']"
    assert_select "input[name='session_estimation_request[hourly_rate]'][value='40']"
    assert_select "select[name='session_estimation_request[tier]'] option", count: 4
    assert_select "select[name='session_estimation_request[tier]'] option[value='']", text: "Auto"
    assert_select "button[type=submit]", text: /Send/
    assert_select "[data-panel='memory']"
    assert_select "[data-form-loading-target='statusPanel'][aria-live='polite']"
    assert_select "header nav a[href='#{root_path}']", text: "Conversation"
    assert_select "header nav a[href='#{new_estimation_path}']", text: "Single estimate"
    assert_select "header nav a[href='#{estimations_path}']", text: "History"
  end

  test "GET show renders the timeline with the user message, attachments and result" do
    chat_session = ChatSession.create!(
      remote_session_id: "sess-1", turn_count: 2,
      latest_metadata: { "project_name" => "Customer Portal", "assumed_team_size" => 4,
                         "mentioned_technologies" => %w[Rails Stripe], "agreed_scope" => "Invoices and payments" }
    )
    create_turn(chat_session, text: "First message about invoices",
                payload: session_estimation_payload(attachments: [ "brief.pdf" ]))
    create_turn(chat_session, text: "Second message adds Stripe")
    stub_snapshot
    get chat_session_path(chat_session)

    assert_response :success
    assert_select "[data-turn]", count: 2
    assert_select "[data-turn]:first-of-type [data-role='user']", text: /First message about invoices/
    assert_select "[data-turn]:first-of-type [data-attachment]", text: "brief.pdf"
    assert_select "[data-turn]:last-of-type [data-role='user']", text: /Second message adds Stripe/
    assert_select "[data-role='assistant'] [data-stat='total-hours']", text: /298/
    assert_select "[data-panel='memory']" do
      assert_select "[data-memory='project-name']", text: "Customer Portal"
      assert_select "[data-memory='team-size']", text: /4/
      assert_select "[data-memory='technology']", count: 2
      assert_select "[data-memory='scope']", text: /Invoices and payments/
      assert_select "[data-memory='turns']", text: /2 of 6/
      assert_select "[data-memory='tier']", text: /developer/
      assert_select "[data-memory='tier']", text: /default/
    end
    assert_select "form[action='#{chat_session_path(chat_session)}'] button", text: /New conversation/
    assert_select "form[action='#{chat_session_path(chat_session)}'] input[name='_method'][value='delete']"
  end

  test "GET show still renders when the snapshot is unavailable" do
    chat_session = ChatSession.create!(remote_session_id: "sess-1", turn_count: 1)
    stub_request(:get, "#{@base}/sess-1").to_return(status: 500)
    get chat_session_path(chat_session)
    assert_response :success
    assert_select "[data-memory='turns']", text: /1/
  end

  test "GET show with an expired remote session starts over" do
    chat_session = ChatSession.create!(remote_session_id: "sess-1")
    stub_snapshot(status: 404, body: { detail: "nope" })
    get chat_session_path(chat_session)
    assert_redirected_to new_chat_session_path
    assert_equal "The conversation expired. A new one was started.", flash[:alert]
    assert_not ChatSession.exists?(chat_session.id)
  end

  test "GET show with an unknown id redirects to new" do
    get chat_session_path(id: 987_654)
    assert_redirected_to new_chat_session_path
  end

  test "DELETE destroys the remote and local session, keeps estimations and clears the session" do
    create = stub_create
    get new_chat_session_path
    chat_session = ChatSession.order(:id).last
    estimation = create_turn(chat_session, text: "Kept in history")
    delete_stub = stub_request(:delete, "#{@base}/sess-1").to_return(status: 204)

    assert_difference "ChatSession.count", -1 do
      delete chat_session_path(chat_session)
    end

    assert_requested delete_stub, times: 1
    assert_redirected_to new_chat_session_path
    assert_response :see_other
    assert_nil estimation.reload.chat_session_id

    # the Rails session key is gone, so the next visit creates a new conversation
    stub_create(session_id: "sess-2")
    get new_chat_session_path
    assert_requested create, times: 2
    assert_equal "sess-2", ChatSession.order(:id).last.remote_session_id
  end

  test "DELETE tolerates a remote failure" do
    chat_session = ChatSession.create!(remote_session_id: "sess-1")
    stub_request(:delete, "#{@base}/sess-1").to_return(status: 500)
    assert_difference "ChatSession.count", -1 do
      delete chat_session_path(chat_session)
    end
    assert_redirected_to new_chat_session_path
  end

  test "DELETE tolerates a timeout" do
    chat_session = ChatSession.create!(remote_session_id: "sess-1")
    stub_request(:delete, "#{@base}/sess-1").to_timeout
    assert_difference "ChatSession.count", -1 do
      delete chat_session_path(chat_session)
    end
  end

  test "DELETE with an unknown id just redirects to new" do
    delete chat_session_path(id: 987_654)
    assert_redirected_to new_chat_session_path
  end
end
