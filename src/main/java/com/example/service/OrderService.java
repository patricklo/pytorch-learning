package com.example.service;

import com.example.dto.AmendOrderRequest;
import com.example.dto.AmendOrderResponse;
import com.example.entity.ClientOrder;

import java.util.List;
import java.util.Optional;

public interface OrderService {

    /**
     * Amends a ClientOrder with exclusive locking
     * @param request The amendment request
     * @return AmendOrderResponse with the result
     */
    AmendOrderResponse amendOrder(AmendOrderRequest request);

    /**
     * Amends a ClientOrder using an existing lock token
     * @param request The amendment request
     * @param lockToken The lock token
     * @return AmendOrderResponse with the result
     */
    AmendOrderResponse amendOrderWithLock(AmendOrderRequest request, String lockToken);

    /**
     * Gets a ClientOrder by ID
     * @param orderId The order ID
     * @return Optional containing the order if found
     */
    Optional<ClientOrder> getOrderById(Long orderId);

    /**
     * Gets a ClientOrder by order number
     * @param orderNumber The order number
     * @return Optional containing the order if found
     */
    Optional<ClientOrder> getOrderByNumber(String orderNumber);

    /**
     * Gets all orders for a client
     * @param clientId The client ID
     * @return List of orders
     */
    List<ClientOrder> getOrdersByClientId(String clientId);

    /**
     * Gets all orders with a specific status
     * @param status The status
     * @return List of orders
     */
    List<ClientOrder> getOrdersByStatus(String status);

    /**
     * Creates a new ClientOrder
     * @param order The order to create
     * @return The created order
     */
    ClientOrder createOrder(ClientOrder order);

    /**
     * Checks if an order can be amended (not locked by another user)
     * @param orderId The order ID
     * @param userId The user ID
     * @return true if order can be amended, false otherwise
     */
    boolean canAmendOrder(Long orderId, String userId);
} 